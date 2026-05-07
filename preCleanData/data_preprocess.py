import os
import pandas as pd
import numpy as np

# --- 1. 設定目標參數 ---
DATA_FOLDER = '../historydata'
OUTPUT_FILE = 'clean_yunlin_air_quality_full.csv'

# 解決政府新舊資料中英混用的問題，「台/臺」、「溼/濕」
STATION_MAPPING = {
    '斗六': 'Douliu', 'Douliu': 'Douliu',
    '崙背': 'Lunbei', 'Lunbei': 'Lunbei',
    '臺西': 'Taixi', '台西': 'Taixi', 'Taixi': 'Taixi',
    '麥寮': 'Mailiao', 'Mailiao': 'Mailiao'
}

ITEM_MAPPING = {
    'PM2.5': 'PM2.5', '細懸浮微粒': 'PM2.5',
    'AMB_TEMP': 'AMB_TEMP', '溫度': 'AMB_TEMP', '環境溫度': 'AMB_TEMP',
    'RH': 'RH', '相對溼度': 'RH', '相對濕度': 'RH',
    'WD_HR': 'WD_HR', '風向': 'WD_HR',
    'WS_HR': 'WS_HR', '風速': 'WS_HR'
}

TARGET_STATIONS_ENG = ['Douliu', 'Lunbei', 'Taixi', 'Mailiao']
TARGET_ITEMS_ENG = ['PM2.5', 'AMB_TEMP', 'RH', 'WD_HR', 'WS_HR']


def process_data():

    all_files = []
    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            if file.lower().endswith('.csv'):
                all_files.append(os.path.join(root, file))

    if not all_files:
        print("找不到 CSV 檔案")
        return

    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding='Big5')
        except Exception as e:
            continue

        # 統一名稱 (把所有中文全部強轉成標準英文)
        if 'sitename' in df.columns:
            df['sitename'] = df['sitename'].map(STATION_MAPPING).fillna(df['sitename'])
        else:
            continue  # 如果連測站欄位都沒有就跳過

        # 尋找測項名稱 (優先找英文欄，沒有就找中文欄)
        if 'itemengname' in df.columns and df['itemengname'].notna().any():
            df['standard_item'] = df['itemengname'].map(ITEM_MAPPING)
        elif 'itemname' in df.columns:
            df['standard_item'] = df['itemname'].map(ITEM_MAPPING)
        else:
            continue

        # 如果英文欄裡面有缺漏，用中文欄補上再 map 一次
        if 'itemname' in df.columns:
            df['standard_item'] = df['standard_item'].fillna(df['itemname'].map(ITEM_MAPPING))

        # 篩選與暫存
        mask = (df['sitename'].isin(TARGET_STATIONS_ENG)) & (df['standard_item'].isin(TARGET_ITEMS_ENG))
        filtered = df[mask].copy()

        if not filtered.empty:
            df_list.append(filtered)
            print(f"📥 讀取成功: {os.path.basename(file)} (成功救回 {len(filtered)} 筆紀錄)")
        else:
            print(f"⚠️ {os.path.basename(file)} 找不到符合條件的資料。")

    if not df_list:
        print("❌ 所有檔案皆未萃取出有效資料。")
        return

    # 合併所有資料
    full_df = pd.concat(df_list, ignore_index=True)

    # 轉換樞紐分析表
    print("\n🧹 正在清洗與重組樞紐分析表...")
    full_df['concentration'] = pd.to_numeric(full_df['concentration'], errors='coerce')

    pivot_df = full_df.pivot_table(
        index=['monitordate', 'sitename'],
        columns='standard_item',
        values='concentration'
    ).reset_index()

    # 填補缺漏值
    print("🩹 正在填補斷線缺漏值...")
    pivot_df['monitordate'] = pd.to_datetime(pivot_df['monitordate'])
    pivot_df = pivot_df.sort_values(by=['sitename', 'monitordate'])

    for item in TARGET_ITEMS_ENG:
        if item in pivot_df.columns:
            pivot_df[item] = pivot_df.groupby('sitename')[item].transform(
                lambda x: x.interpolate(method='linear').bfill().ffill())

    # 將風拆解為 X/Y 向量
    print("🌪️ 正在執行特徵工程：解析風速與風向...")
    if 'WD_HR' in pivot_df.columns and 'WS_HR' in pivot_df.columns:
        rad = np.deg2rad(pivot_df['WD_HR'])
        pivot_df['Wind_X'] = pivot_df['WS_HR'] * np.cos(rad)
        pivot_df['Wind_Y'] = pivot_df['WS_HR'] * np.sin(rad)
        pivot_df = pivot_df.drop(columns=['WD_HR'])
    else:
        print("⚠️ 警告：這批資料中缺少風向(WD_HR)或風速(WS_HR)。")

    # 儲存
    pivot_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ 煉金完成！24個月的完整資料已儲存為：{OUTPUT_FILE}")
    print(f"📊 總資料筆數預計突破 7萬筆！實際筆數：{len(pivot_df)} 筆")


if __name__ == "__main__":
    process_data()