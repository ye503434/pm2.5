import os
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from torch.utils.data import Dataset, DataLoader

class STGNNDataset(Dataset):
    def __init__(self, data, seq_len=24):
        """
        data shape: [Time, Num_Nodes, Features]
        """
        self.data = data
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len
        
    def __getitem__(self, idx):
        # 取過去 24 小時的特徵作為 X
        x = self.data[idx : idx + self.seq_len] 
        # 取第 25 小時的 PM2.5 (index 0) 作為 Y
        y = self.data[idx + self.seq_len, :, 0] 
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class STGNN(nn.Module):
    def __init__(self, num_nodes, in_features, gcn_out, lstm_hidden):
        super(STGNN, self).__init__()
        self.num_nodes = num_nodes
        
        # 空間層：使用 GCNConv 提取節點間的空間特徵
        self.gcn = GCNConv(in_features, gcn_out)
        
        # 時間層：將 GCN 的輸出送入 nn.LSTM 提取時間序列特徵
        # GCN 輸出後，每個節點有一個長度為 gcn_out 的向量。
        # 將所有節點的特徵攤平 (num_nodes * gcn_out) 作為 LSTM 的輸入
        self.lstm = nn.LSTM(input_size=num_nodes * gcn_out, hidden_size=lstm_hidden, batch_first=True)
        
        # 輸出層：使用 nn.Linear，一次輸出 4 個測站的 PM2.5 預測值
        self.linear = nn.Linear(lstm_hidden, num_nodes)

    def forward(self, x, edge_index):
        # 預期輸入形狀 x: [Batch, Seq_len, Num_Nodes, Features]
        batch_size, seq_len, num_nodes, features = x.shape
        
        # 將 Batch 與 Seq_len 攤平，方便一次送入 GCN [Batch * Seq_len * Num_Nodes, Features]
        x_flat = x.view(batch_size * seq_len * num_nodes, features)
        
        # 動態產生對應 batch 大小的 edge_index
        num_graphs = batch_size * seq_len
        edge_index_batched = torch.cat(
            [edge_index + i * num_nodes for i in range(num_graphs)], dim=1
        )
        
        # GCN 空間特徵提取
        gcn_out = self.gcn(x_flat, edge_index_batched)
        gcn_out = torch.relu(gcn_out)
        
        # 重塑回 [Batch, Seq_len, Num_Nodes * gcn_out] 以送入 LSTM
        gcn_out = gcn_out.view(batch_size, seq_len, num_nodes * self.gcn.out_channels)
        
        # LSTM 時間特徵提取
        lstm_out, _ = self.lstm(gcn_out)
        
        # 取最後一個時間步的輸出
        last_out = lstm_out[:, -1, :] # [Batch, lstm_hidden]
        
        # 線性層預測 4 個測站的 PM2.5
        out = self.linear(last_out) # [Batch, Num_Nodes]
        return out

def main():
    # ==========================
    # 1. 參數與路徑設定
    # ==========================
    DATA_PATH = '../preCleanData/clean_yunlin_air_quality_full.csv'
    OUTPUT_DIR = '.'
    SEQ_LEN = 24
    BATCH_SIZE = 32
    EPOCHS = 60
    LR = 0.0005
    
    # 檢查 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 取得檔案絕對路徑 (相對於本腳本)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_full_path = os.path.normpath(os.path.join(script_dir, DATA_PATH))
    
    # 防呆：檢查檔案是否存在
    if not os.path.exists(data_full_path):
        raise FileNotFoundError(f"找不到資料檔案：{data_full_path}")
        
    # ==========================
    # 2. 資料載入與前處理
    # ==========================
    print("Loading data...")
    df = pd.read_csv(data_full_path)
    
    en_names = ['Douliu', 'Lunbei', 'Taixi', 'Mailiao']
    zh_names = ['斗六', '崙背', '臺西', '麥寮']
    site_mapping = dict(zip(en_names, zh_names))
    
    features_cols = ['PM2.5', 'AMB_TEMP', 'RH', 'Wind_X', 'Wind_Y']
    
    df['monitordate'] = pd.to_datetime(df['monitordate'])
    
    # 確認是否包含所有需要的特徵
    missing_cols = [col for col in features_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"資料集中缺少必要特徵: {missing_cols}")
        
    # 整理各站點資料並對齊時間
    all_times = df['monitordate'].drop_duplicates().sort_values()
    data_dict = {}
    for site in en_names:
        site_df = df[df['sitename'] == site].sort_values('monitordate')
        site_df.set_index('monitordate', inplace=True)
        # 對齊時間並使用 ffill/bfill 處理缺失值
        data_dict[site] = site_df[features_cols].reindex(all_times).ffill().bfill()
        
    # ==========================
    # 3. 建立正規化模型 
    # ==========================
    # 僅使用前 80% 訓練資料 fit，避免資訊洩漏
    train_size_idx = int(len(all_times) * 0.8)
    full_data = np.zeros((len(all_times), len(en_names), len(features_cols)))
    scalers = {}
    
    print("Normalizing data...")
    for i, site in enumerate(en_names):
        scaler = MinMaxScaler()
        site_data = data_dict[site].values
        
        # 僅用 train data 進行 fit
        scaler.fit(site_data[:train_size_idx])
        scaled_site_data = scaler.transform(site_data)
        
        scalers[site] = scaler
        full_data[:, i, :] = scaled_site_data
        
    # ==========================
    # 4. 建立 Dataset 與 DataLoader
    # ==========================
    dataset = STGNNDataset(full_data, seq_len=SEQ_LEN)
    
    # 因為 dataset 長度 = len(all_times) - SEQ_LEN，重新計算 split index
    dataset_train_size = int(len(dataset) * 0.8)
    
    train_dataset = torch.utils.data.Subset(dataset, range(0, dataset_train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(dataset_train_size, len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ==========================
    # 5. 建立圖結構 (Graph Topology)
    # ==========================
    # 0:斗六, 1:崙背, 2:臺西, 3:麥寮
    # 拓撲關係：斗六-崙背、崙背-臺西、崙背-麥寮、臺西-麥寮
    # 因為需要雙向邊，手動列出所有 [源節點, 目標節點]
    edge_index = torch.tensor([
        [0, 1, 1, 2, 1, 3, 2, 3], # source nodes
        [1, 0, 2, 1, 3, 1, 3, 2]  # target nodes
    ], dtype=torch.long).to(device)
    
    # ==========================
    # 6. 初始化 STGNN 模型
    # ==========================
    num_nodes = len(en_names)
    in_features = len(features_cols)
    gcn_out = 64
    lstm_hidden = 128
    
    model = STGNN(num_nodes=num_nodes, in_features=in_features, gcn_out=gcn_out, lstm_hidden=lstm_hidden).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # ==========================
    # 7. 訓練迴圈
    # ==========================
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            out = model(x_batch, edge_index)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # 驗證迴圈
        model.eval()
        all_preds = []
        all_trues = []
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch, edge_index)
                
                loss = criterion(out, y_batch)
                val_loss += loss.item()
                
                all_preds.append(out.cpu().numpy())
                all_trues.append(y_batch.cpu().numpy())
                
        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)
        
        # 反正規化並計算整體 MAE
        val_mae = 0
        for i, site in enumerate(en_names):
            scaler = scalers[site]
            
            # Dummy array 供反正規化使用 (因為原本 MinMaxScaler fit 在 5 維度上)
            dummy_pred = np.zeros((len(all_preds), in_features))
            dummy_pred[:, 0] = all_preds[:, i]
            denorm_pred = scaler.inverse_transform(dummy_pred)[:, 0]
            
            dummy_true = np.zeros((len(all_trues), in_features))
            dummy_true[:, 0] = all_trues[:, i]
            denorm_true = scaler.inverse_transform(dummy_true)[:, 0]
            
            site_mae = np.mean(np.abs(denorm_pred - denorm_true))
            val_mae += site_mae
            
        val_mae /= num_nodes
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {total_loss/len(train_loader):.4f}, Val MSE: {val_loss/len(val_loader):.4f}, Val MAE: {val_mae:.4f}")
        
    # ==========================
    # 8. 儲存模型與 Scalers
    # ==========================
    model_path = os.path.join(script_dir, OUTPUT_DIR, 'stgnn_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {os.path.normpath(model_path)}")
    
    for en_site, zh_site in site_mapping.items():
        scaler_path = os.path.join(script_dir, OUTPUT_DIR, f'stgnn_scaler_{zh_site}.pkl')
        joblib.dump(scalers[en_site], scaler_path)
        print(f"Scaler saved to {os.path.normpath(scaler_path)}")

if __name__ == '__main__':
    main()
