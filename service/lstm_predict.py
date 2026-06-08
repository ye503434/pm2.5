import torch
import torch.nn as nn
import numpy as np
import pickle
import os

# ================================================================
# 升級版 LSTM 推論程式 (支援 5 維度空間特徵)
# ================================================================
#
# 使用方法:
#   from predict import predict_pm25
#   result = predict_pm25(hist_pm25, hist_temp, hist_hum, hist_wind_x, hist_wind_y, station='麥寮')
#
# 參數說明 (所有 list 長度必須嚴格等於 24):
#   hist_pm25     : list，過去 24 小時的 PM2.5 數值
#   hist_temp     : list，過去 24 小時的溫度數值
#   hist_hum      : list，過去 24 小時的濕度數值
#   hist_wind_x   : list，過去 24 小時的風向 X 向量 (東西向)
#   hist_wind_y   : list，過去 24 小時的風向 Y 向量 (南北向)
#   station       : str，站名 ('斗六', '崙背', '臺西', '麥寮')
#
# 回傳值:
#   float，預測的下一小時 PM2.5 數值 (發生錯誤時回傳 None)
# ================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))

# 必須與 train.py 中的特徵數量與順序完全一致！
FEATURES_COUNT = 5
SEQ_LENGTH = 24


# 模型架構定義 (修復致命傷一：input_size 改為 5)
class LSTMModel(nn.Module):
    def __init__(self, input_size=FEATURES_COUNT, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# 預先載入所有模型與正規化器 (避免每次預測都要重新讀取檔案)
SUPPORTED_STATIONS = ['斗六', '崙背', '臺西', '麥寮']
models = {}
scalers = {}

print("🔄 正在初始化 LSTM 預測引擎...")
for station in SUPPORTED_STATIONS:
    model_path = os.path.join(current_dir, f'lstm_{station}.pth')
    scaler_path = os.path.join(current_dir, f'scaler_{station}.pkl')

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        # 保命符一：強制使用 CPU 載入
        model = LSTMModel()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # 切換到推論模式
        models[station] = model

        with open(scaler_path, 'rb') as f:
            scalers[station] = pickle.load(f)
        print(f"  ✅ 成功載入 {station} 的大腦與轉換器")
    else:
        print(f"  ⚠️ 找不到 {station} 的模型檔案，請確認是否已執行 train.py")


# 核心預測函數
def predict_pm25(history_pm25: list, history_temp: list, history_humidity: list,
                 history_wind_x: list, history_wind_y: list, station: str = '斗六'):
    if station not in models:
        print(f"❌ 錯誤：找不到 {station} 的模型。")
        return None

    # 嚴格的長度防呆機制 (防止 API 斷線少給資料)
    lists_to_check = [history_pm25, history_temp, history_humidity, history_wind_x, history_wind_y]
    for i, lst in enumerate(lists_to_check):
        if len(lst) != SEQ_LENGTH:
            print(f"❌ 錯誤：傳入的歷史資料長度不對！必須剛好 {SEQ_LENGTH} 筆，但第 {i + 1} 個特徵只有 {len(lst)} 筆。")
            return None

    model = models[station]
    scaler = scalers[station]

    # 將五個特徵依照 train.py 的順序打包
    # 順序必須是: ['PM2.5', 'AMB_TEMP', 'RH', 'Wind_X', 'Wind_Y']
    features = np.array([
        [p, t, h, wx, wy] for p, t, h, wx, wy in zip(
            history_pm25, history_temp, history_humidity, history_wind_x, history_wind_y
        )
    ], dtype=np.float32)

    # 進行正規化 (Scaler 現在預期看到 5 個欄位)
    scaled_features = scaler.transform(features)

    # 轉換成 PyTorch Tensor，並加上 Batch 維度 -> 形狀變成 (1, 24, 5)
    X = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)

    # 封印反向傳播 (省記憶體)
    with torch.no_grad():
        output_scaled = model(X).item()

    # 安全的工業級反正規化 (Inverse Scaling)
    # 建立一個 (1, 5) 的假陣列，把預測出來的 PM2.5 放進 index 0 的位置
    dummy_array = np.zeros((1, FEATURES_COUNT))
    dummy_array[0, 0] = output_scaled

    # 使用 scaler 原生的方法還原
    real_prediction = scaler.inverse_transform(dummy_array)[0, 0]

    # 防止出現不合理的負數 PM2.5 (這在神經網路有時會發生)
    final_pm25 = max(0.0, round(real_prediction, 1))

    return final_pm25


if __name__ == "__main__":
    print("\n🧪 進行本地端推論測試...")
    # 捏造 24 小時的假資料來測試
    fake_pm25 = [20.0] * 24
    fake_temp = [25.0] * 24
    fake_hum = [80.0] * 24
    fake_wind_x = [1.5] * 24
    fake_wind_y = [-0.5] * 24

    test_station = '麥寮'
    result = predict_pm25(fake_pm25, fake_temp, fake_hum, fake_wind_x, fake_wind_y, station=test_station)

    if result is not None:
        print(f"🎉 測試成功！根據假資料，{test_station} 下一小時的預測 PM2.5 為：{result} μg/m³")