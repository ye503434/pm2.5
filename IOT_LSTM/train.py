import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# ==========================================
# 1. 超參數與全域設定 (Hyperparameters)
# ==========================================
DATA_FILE = '../preCleanData/clean_yunlin_air_quality_full.csv'
SEQ_LENGTH = 24  # 拿過去 24 小時的資料
PREDICT_AHEAD = 1  # 預測未來 1 小時
BATCH_SIZE = 64
EPOCHS = 50  # 訓練回合數
LEARNING_RATE = 0.001

# 特徵順序非常重要！PM2.5 必須放在第一個 (index 0)
FEATURES = ['PM2.5', 'AMB_TEMP', 'RH', 'Wind_X', 'Wind_Y']
STATIONS = ['斗六', '崙背', '臺西', '麥寮']

# 自動切換 GPU/CPU (如果有 NVIDIA 顯卡會自動加速)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 使用運算設備: {device}")


# ==========================================
# 2. LSTM 模型架構 (修正 input_size=5)
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=len(FEATURES), hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        # batch_first=True 代表輸入維度是 (Batch, Sequence, Features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        # 壓縮成 1 維，預測 PM2.5 數值
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # out: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x)
        # 我們只取最後一個時間點的輸出作為預測結果
        return self.fc(out[:, -1, :])


# ==========================================
# 3. 建立時間序列視窗 (Sliding Window)
# ==========================================
def create_sequences(data, seq_length):
    xs, ys = [], []
    # 預留未來的 1 小時作為答案 (y)
    for i in range(len(data) - seq_length - PREDICT_AHEAD + 1):
        x = data[i:(i + seq_length)]
        # target 是下一個小時的 PM2.5 (即特徵矩陣的 index 0)
        y = data[i + seq_length + PREDICT_AHEAD - 1][0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# ==========================================
# 4. 主訓練流程
# ==========================================
def train_station_models():
    print("讀取黃金訓練資料中...")
    df = pd.read_csv(DATA_FILE)
    # 確保時間排序正確，這對時間序列至關重要
    df['monitordate'] = pd.to_datetime(df['monitordate'])
    df = df.sort_values(by=['sitename', 'monitordate'])

    for station in STATIONS:
        print(f"\n{'=' * 40}")
        print(f"🏭 開始訓練測站：{station}")
        print(f"{'=' * 40}")

        # 1. 萃取該測站資料
        # 處理新舊英文名稱轉換
        station_eng_map = {'斗六': 'Douliu', '崙背': 'Lunbei', '臺西': 'Taixi', '麥寮': 'Mailiao'}
        station_data = df[df['sitename'] == station_eng_map[station]][FEATURES].values

        if len(station_data) < 1000:
            print(f"⚠️ {station} 資料量不足，跳過訓練！")
            continue

        # 2. 正規化 (Data Scaling)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(station_data)

        # 儲存 Scaler 給預測時使用
        with open(f'scaler_{station}.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        # 3. 切割 Time Windows
        X, y = create_sequences(scaled_data, SEQ_LENGTH)

        # 4. 嚴格的時間序列 Train/Test Split (前 80% 訓練，後 20% 驗證)
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]

        # 轉成 PyTorch Tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

        # 建立 DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 5. 初始化模型、損失函數、優化器
        model = LSTMModel().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 6. 開始訓練迴圈
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 每 10 個 Epoch 印出一次進度
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test_preds = model(X_test_t)
                    test_loss = criterion(test_preds, y_test_t).item()
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss(MSE): {train_loss / len(train_loader):.4f} | Test Loss: {test_loss:.4f}")

        # 7. 最終評估 (還原數值，計算人類看得懂的 MAE 誤差)
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_t).cpu().numpy()
            y_test_actual = y_test_t.cpu().numpy()

            # 建立假的 5 維度陣列來安全地反正規化
            dummy_preds = np.zeros((len(test_preds), len(FEATURES)))
            dummy_preds[:, 0] = test_preds[:, 0]
            real_preds = scaler.inverse_transform(dummy_preds)[:, 0]

            dummy_actuals = np.zeros((len(y_test_actual), len(FEATURES)))
            dummy_actuals[:, 0] = y_test_actual[:, 0]
            real_actuals = scaler.inverse_transform(dummy_actuals)[:, 0]

            # 計算平均絕對誤差 (MAE)
            mae = np.mean(np.abs(real_preds - real_actuals))
            print(f"\n🎯 {station} 訓練完成！")
            print(f"📊 測試集平均誤差 (MAE): 預測結果平均正負相差 {mae:.2f} μg/m³")

        # 8. 儲存模型權重
        torch.save(model.state_dict(), f'lstm_{station}.pth')
        print(f"💾 模型已儲存為 lstm_{station}.pth\n")


if __name__ == "__main__":
    train_station_models()