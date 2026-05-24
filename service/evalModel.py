import os
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# 1. 重構模型架構 (必須與訓練完全一致)
# ==========================================

# --- LSTM 模型架構 ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# --- STGNN 模型架構 ---
class STGNN(nn.Module):
    def __init__(self, num_nodes=4, in_features=5, gcn_out=32, lstm_hidden=64):
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
        return self.linear(last_out)


# ==========================================
# 2. 主要評估管線流程
# ==========================================
def main():
    DATA_PATH = '../preCleanData/clean_yunlin_air_quality_full.csv'
    SEQ_LEN = 24
    device = torch.device('cpu')  # 評估階段統一使用 CPU 確保環境相容性

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_full_path = os.path.normpath(os.path.join(script_dir, DATA_PATH))

    print("🔄 步驟 1: 正在讀取並對齊四個測站的歷史資料...")
    df = pd.read_csv(data_full_path)
    df['monitordate'] = pd.to_datetime(df['monitordate'])

    en_names = ['Douliu', 'Lunbei', 'Taixi', 'Mailiao']
    zh_names = ['斗六', '崙背', '臺西', '麥寮']
    features_cols = ['PM2.5', 'AMB_TEMP', 'RH', 'Wind_X', 'Wind_Y']

    all_times = df['monitordate'].drop_duplicates().sort_values()

    # 建立時序對齊的三維數據矩陣 [Time, Nodes, Features]
    full_data = np.zeros((len(all_times), len(en_names), len(features_cols)))
    scalers = {}

    for i, (en_site, zh_site) in enumerate(zip(en_names, zh_names)):
        site_df = df[df['sitename'] == en_site].sort_values('monitordate').set_index('monitordate')
        aligned_df = site_df[features_cols].reindex(all_times).ffill().bfill()

        # 載入當初訓練儲存的 Scaler 進行正規化
        scaler_path = os.path.join(script_dir, f'stgnn_scaler_{zh_site}.pkl')
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(script_dir, f'scaler_{zh_site}.pkl')  # 彈性相容另一套命名

        with open(scaler_path, 'rb') as f:
            scaler = joblib.load(f) if scaler_path.endswith('.pkl') else pickle.load(f)

        scalers[zh_site] = scaler
        full_data[:, i, :] = scaler.transform(aligned_df.values)

    # 建立滑動視窗測試集 (嚴格切分後 20% 的時間序列)
    total_windows = len(full_data) - SEQ_LEN
    test_start_idx = int(total_windows * 0.8)

    X_test_list, Y_test_list = [], []
    for idx in range(test_start_idx, total_windows):
        X_test_list.append(full_data[idx: idx + SEQ_LEN])
        Y_test_list.append(full_data[idx + SEQ_LEN, :, 0])  # Index 0 為 PM2.5 真實值

    X_test_stgnn = np.array(X_test_list)  # [Batch, Seq, Nodes, Features]
    Y_test_real = np.array(Y_test_list)  # [Batch, Nodes]

    print(f"📊 成功建立獨立測試集！總評估樣本數: {len(X_test_stgnn)} 小時")

    # ==========================================
    # 3. 執行 STGNN 模型推論
    # ==========================================
    print("\n🔮 步驟 2: 正在執行 STGNN 全局聯合預測...")
    model_stgnn = STGNN().to(device)
    stgnn_path = os.path.join(script_dir, 'stgnn_model.pth')
    model_stgnn.load_state_dict(torch.load(stgnn_path, map_location=device, weights_only=True))
    model_stgnn.eval()

    edge_index = torch.tensor([
        [0, 1, 1, 2, 1, 3, 2, 3],
        [1, 0, 2, 1, 3, 1, 3, 2]
    ], dtype=torch.long).to(device)

    with torch.no_grad():
        stgnn_inputs = torch.tensor(X_test_stgnn, dtype=torch.float32).to(device)
        stgnn_outputs = model_stgnn(stgnn_inputs, edge_index).cpu().numpy()  # [Batch, Nodes]

    # ==========================================
    # 4. 執行 LSTM 模型推論 (分站獨立進行)
    # ==========================================
    print("🔮 步驟 3: 正在執行 LSTM 各站獨立預測...")
    lstm_outputs = np.zeros_like(stgnn_outputs)  # 用來存放 4 個站的 LSTM 預測結果

    for i, zh_site in enumerate(zh_names):
        model_lstm = LSTMModel().to(device)
        lstm_path = os.path.join(script_dir, f'lstm_{zh_site}.pth')
        model_lstm.load_state_dict(torch.load(lstm_path, map_location=device, weights_only=True))
        model_lstm.eval()

        # 從對齊的矩陣中抽取出該特定站點的資料 [Batch, Seq, Features]
        X_test_lstm = X_test_stgnn[:, :, i, :]

        with torch.no_grad():
            lstm_inputs = torch.tensor(X_test_lstm, dtype=torch.float32).to(device)
            lstm_outputs[:, i] = model_lstm(lstm_inputs).cpu().numpy().flatten()

    # ==========================================
    # 5. 反正規化還原真實值與數據計算
    # ==========================================
    print("🧮 步驟 4: 正在進行工業級反正規化並計算誤差指標...")

    metrics_summary = {}

    for i, zh_site in enumerate(zh_names):
        scaler = scalers[zh_site]

        # 還原真實觀測答案值
        dummy = np.zeros((len(Y_test_real), len(features_cols)))
        dummy[:, 0] = Y_test_real[:, i]
        real_y = scaler.inverse_transform(dummy)[:, 0]

        # 還原 STGNN 預測值
        dummy[:, 0] = stgnn_outputs[:, i]
        pred_stgnn = scaler.inverse_transform(dummy)[:, 0]
        pred_stgnn = np.clip(pred_stgnn, 0, None)  # 限制合理邊界

        # 還原 LSTM 預測值
        dummy[:, 0] = lstm_outputs[:, i]
        pred_lstm = scaler.inverse_transform(dummy)[:, 0]
        pred_lstm = np.clip(pred_lstm, 0, None)

        # 計算指標
        mae_lstm = mean_absolute_error(real_y, pred_lstm)
        rmse_lstm = np.sqrt(mean_squared_error(real_y, pred_lstm))

        mae_stgnn = mean_absolute_error(real_y, pred_stgnn)
        rmse_stgnn = np.sqrt(mean_squared_error(real_y, pred_stgnn))

        metrics_summary[zh_site] = {
            'LSTM_MAE': mae_lstm, 'LSTM_RMSE': rmse_lstm,
            'STGNN_MAE': mae_stgnn, 'STGNN_RMSE': rmse_stgnn
        }

    # ==========================================
    # 6. 終端機美化輸出 Markdown 表格
    # ==========================================
    print("\n" + "=" * 50)
    print("📊 空氣品質模型預測表現對比報告 (獨立測試集)")
    print("=" * 50)
    print("| 測站名稱 | LSTM MAE | STGNN MAE | 領先幅度 (MAE) | LSTM RMSE | STGNN RMSE |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: |")

    avg_l_mae, avg_s_mae, avg_l_rmse, avg_s_rmse = [], [], [], []

    for site, m in metrics_summary.items():
        diff = m['LSTM_MAE'] - m['STGNN_MAE']
        lead_str = f"STGNN 贏 {diff:.2f}" if diff > 0 else f"LSTM 贏 {abs(diff):.2f}"
        print(
            f"| {site} | {m['LSTM_MAE']:.2f} | {m['STGNN_MAE']:.2f} | {lead_str} | {m['LSTM_RMSE']:.2f} | {m['STGNN_RMSE']:.2f} |")

        avg_l_mae.append(m['LSTM_MAE'])
        avg_s_mae.append(m['STGNN_MAE'])
        avg_l_rmse.append(m['LSTM_RMSE'])
        avg_s_rmse.append(m['STGNN_RMSE'])

    print("| **整體平均** | **{:.2f}** | **{:.2f}** | **STGNN 贏 {:.2f}** | **{:.2f}** | **{:.2f}** |".format(
        np.mean(avg_l_mae), np.mean(avg_s_mae), np.mean(avg_l_mae) - np.mean(avg_s_mae), np.mean(avg_l_rmse),
        np.mean(avg_s_rmse)
    ))
    print("=" * 50)

    # ==========================================
    # 7. 視覺化分組長條圖繪製
    # ==========================================
    print("\n🎨 步驟 5: 正在繪製並儲存分組長條圖...")

    # 為了防止在某些無圖形畫面的 Linux 伺服器上報錯，使用英文字型防止中文方塊
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x_indexes = np.arange(len(zh_names))
    bar_width = 0.35

    # 圖一：MAE 對比
    ax1.bar(x_indexes - bar_width / 2, avg_l_mae, bar_width, label='LSTM', color='#ff5252', alpha=0.8)
    ax1.bar(x_indexes + bar_width / 2, avg_s_mae, bar_width, label='STGNN', color='#00bcd4', alpha=0.8)
    ax1.set_title('Model MAE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_indexes)
    ax1.set_xticklabels(['Douliu', 'Lunbei', 'Taixi', 'Mailiao'])
    ax1.set_ylabel('MAE (ug/m3)')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend()

    # 圖二：RMSE 對比
    ax2.bar(x_indexes - bar_width / 2, avg_l_rmse, bar_width, label='LSTM', color='#ff7a7a', alpha=0.8)
    ax2.bar(x_indexes + bar_width / 2, avg_s_rmse, bar_width, label='STGNN', color='#4dd0e1', alpha=0.8)
    ax2.set_title('Model RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_indexes)
    ax2.set_xticklabels(['Douliu', 'Lunbei', 'Taixi', 'Mailiao'])
    ax2.set_ylabel('RMSE (ug/m3)')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    chart_output_path = os.path.join(script_dir, 'model_comparison.png')
    plt.savefig(chart_output_path, dpi=300)
    print(f"🎉 評估報告與圖表製作完成！對比圖已成功儲存至: {chart_output_path}")


if __name__ == '__main__':
    main()