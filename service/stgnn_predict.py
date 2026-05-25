import os
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import numpy as np
import joblib


# ==========================================
# 1. 網路架構 (必須與訓練時完全一致)
# ==========================================
class STGNN(nn.Module):
    def __init__(self, num_nodes, in_features, gcn_out, lstm_hidden):
        super(STGNN, self).__init__()
        self.num_nodes = num_nodes
        self.gcn = GCNConv(in_features, gcn_out)
        self.lstm = nn.LSTM(input_size=num_nodes * gcn_out, hidden_size=lstm_hidden, batch_first=True)
        self.linear = nn.Linear(lstm_hidden, num_nodes)

    def forward(self, x, edge_index):
        batch_size, seq_len, num_nodes, features = x.shape
        x_flat = x.view(batch_size * seq_len * num_nodes, features)

        num_graphs = batch_size * seq_len
        edge_index_batched = torch.cat(
            [edge_index + i * num_nodes for i in range(num_graphs)], dim=1
        )

        gcn_out = self.gcn(x_flat, edge_index_batched)
        gcn_out = torch.relu(gcn_out)
        gcn_out = gcn_out.view(batch_size, seq_len, num_nodes * self.gcn.out_channels)

        lstm_out, _ = self.lstm(gcn_out)
        last_out = lstm_out[:, -1, :]
        out = self.linear(last_out)
        return out


# ==========================================
# 2. 推論函式
# ==========================================
def predict_stgnn(recent_data):
    """
    執行 STGNN 模型的推論
    :param recent_data: dict, 包含 4 個測站過去 24 小時的 5 維特徵
    :return: dict, 4 個測站的 PM2.5 預測值
    """
    # 站點名稱與特徵鍵值 (順序必須與訓練時嚴格一致)
    zh_names = ['斗六', '崙背', '臺西', '麥寮']
    features_keys = ['pm25', 'temp', 'rh', 'wind_x', 'wind_y']

    # 取得當前腳本路徑，確保能正確找到權重檔
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'stgnn_model.pth')

    # 自動偵測運行設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 步驟 A: 初始化並載入模型權重
    model = STGNN(num_nodes=4, in_features=5, gcn_out=64, lstm_hidden=128).to(device)
    # 使用 weights_only=True 提升載入安全性 (PyTorch 建議)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # 切換到評估模式

    # 步驟 B: 資料提取與正規化 (建立 shape 為 [24, 4, 5] 的陣列)
    x_data = np.zeros((24, 4, 5))
    scalers = {}

    for i, site in enumerate(zh_names):
        # 載入該站點對應的 Scaler
        scaler_path = os.path.join(script_dir, f'stgnn_scaler_{site}.pkl')
        scalers[site] = joblib.load(scaler_path)

        # 提取傳入字典中該站的 5 個特徵，並組合成 [24, 5] 的二維陣列
        site_features = np.zeros((24, 5))
        for j, f_key in enumerate(features_keys):
            site_features[:, j] = recent_data[site][f_key]

        # 將資料送入 Scaler 正規化，然後存入 x_data 對應的站點維度
        x_data[:, i, :] = scalers[site].transform(site_features)

    # 步驟 C: 轉換為 Tensor 並增加 Batch 維度 -> [1, 24, 4, 5]
    x_tensor = torch.tensor(x_data, dtype=torch.float32).unsqueeze(0).to(device)

    # 步驟 D: 建立拓撲關係圖的 edge_index
    edge_index = torch.tensor([
        [0, 1, 1, 2, 1, 3, 2, 3],
        [1, 0, 2, 1, 3, 1, 3, 2]
    ], dtype=torch.long).to(device)

    # 步驟 E: 執行推論
    with torch.no_grad():
        out = model(x_tensor, edge_index)  # 輸出形狀: [1, 4]
        out_np = out.cpu().numpy()[0]  # 攤平為一維陣列: [4]

    # 步驟 F: 反正規化還原為真實 PM2.5 數值
    result = {}
    for i, site in enumerate(zh_names):
        # 建立與訓練時相同特徵數 (5維) 的 Dummy Array
        dummy = np.zeros((1, 5))
        # 將預測結果放在 PM2.5 (index 0) 的位置
        dummy[0, 0] = out_np[i]

        # 反正規化並取出還原後的值
        denorm_val = scalers[site].inverse_transform(dummy)[0, 0]

        # 轉換為標準 float 型態，並四捨五入到小數點第一位，讓結果更易讀
        result[site] = round(float(denorm_val), 1)

    return result


# ==========================================
# 3. 測試區塊 (可選)
# ==========================================
if __name__ == '__main__':
    # 這裡示範如何呼叫 predict_stgnn
    # 產生一組假的測試資料 (結構與你要求的格式一致)
    dummy_recent_data = {
        site: {
            "pm25": np.random.uniform(10, 50, 24).tolist(),
            "temp": np.random.uniform(20, 30, 24).tolist(),
            "rh": np.random.uniform(60, 90, 24).tolist(),
            "wind_x": np.random.uniform(-5, 5, 24).tolist(),
            "wind_y": np.random.uniform(-5, 5, 24).tolist()
        }
        for site in ["斗六", "崙背", "臺西", "麥寮"]
    }

    # 執行推論
    try:
        predictions = predict_stgnn(dummy_recent_data)
        print("====== 預測結果 ======")
        print(predictions)
    except FileNotFoundError as e:
        print(f"錯誤：{e}")
        print("💡 請確認你已經執行過 stgnn_train.py，並且有生成 stgnn_model.pth 和 4 個 scaler_XXX.pkl 檔案！")