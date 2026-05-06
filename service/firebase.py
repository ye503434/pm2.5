import requests
import os
from datetime import datetime
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# --- 1. 系統設定與金鑰讀取 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, '../.env'))
API_KEY = os.getenv("MOENV_API_KEY")
API_URL = f"https://data.moenv.gov.tw/api/v2/aqx_p_145?api_key={API_KEY}&limit=1000&sort=monitordate desc&format=JSON"

# --- 2. 初始化 Firebase ---
cred_path = os.path.join(current_dir, '../firebase-key.json')
if not os.path.exists(cred_path):
    raise FileNotFoundError("找不到 firebase-key.json！")

print("正在連線至 Firebase...")
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
db = firestore.client()

TARGET_STATIONS = ['斗六', '崙背', '臺西', '麥寮']

# --- 3. 主程式：抓取資料並累積 ---
def fetch_and_upload():
    print("正在向環境部抓取最新資料...")
    response = requests.get(API_URL)
    if response.status_code != 200:
        print("❌ 抓取失敗！")
        return

    data = response.json()
    records = data if isinstance(data, list) else data.get('records', [])

    current_data_map = {}
    update_time = ""

    # 解析最新一筆資料
    for item in records:
        sitename = item.get('sitename')
        if sitename in TARGET_STATIONS:
            if sitename not in current_data_map:
                current_data_map[sitename] = {"pm25": 0, "temperature": 0, "humidity": 0}
                update_time = item.get('monitordate')

            item_eng = item.get('itemengname')
            val_str = item.get('concentration')
            try:
                val = float(val_str) if val_str and val_str.lower() != 'x' else 0
            except ValueError:
                val = 0

            if item_eng == 'PM2.5': current_data_map[sitename]["pm25"] = val
            elif item_eng == 'AMB_TEMP': current_data_map[sitename]["temperature"] = val
            elif item_eng == 'RH': current_data_map[sitename]["humidity"] = val

    # 轉換時間格式 (例：2026-05-06 18:00 -> 05/06 18:00)
    try:
        dt_obj = datetime.strptime(update_time, "%Y-%m-%d %H:%M:%S")
        short_time = dt_obj.strftime("%m/%d %H:%M")
    except:
        short_time = update_time[5:16].replace("-", "/") if update_time else "未知時間"

    # --- 核心改變：讀取 Firebase 舊資料來累積 ---
    print("📥 正在從 Firebase 讀取歷史紀錄...")
    doc_ref = db.collection('air_quality').document('latest')
    doc_snap = doc_ref.get()

    if doc_snap.exists:
        old_data = doc_snap.to_dict()
    else:
        old_data = {"stations_data": {}} # 如果雲端是空的，就開一個全新的

    final_upload_data = {
        "status": "success",
        "update_time": update_time,
        "stations_data": {}
    }

    # 將新資料合併到舊歷史中
    for station in TARGET_STATIONS:
        new_current = current_data_map.get(station, {"pm25": 0, "temperature": 0, "humidity": 0})

        # 拿出舊的歷史陣列 (如果沒有就生出空的)
        old_station_data = old_data.get("stations_data", {}).get(station, {})
        history = old_station_data.get("history", {"labels": [], "temperature": [], "pm25": []})

        # 防呆機制：檢查這個時間點是否已經寫入過？(避免你在同一個小時內按太多次執行)
        if len(history["labels"]) > 0 and history["labels"][-1] == short_time:
            print(f"[{station}] {short_time} 資料已存在，更新數值。")
            history["temperature"][-1] = new_current["temperature"]
            history["pm25"][-1] = new_current["pm25"]
        else:
            # 沒寫入過，把最新的真實點加到陣列最後面
            history["labels"].append(short_time)
            history["temperature"].append(new_current["temperature"])
            history["pm25"].append(new_current["pm25"])

        # 限制陣列長度，最多只保留過去 120 筆 (大約 5 天)
        if len(history["labels"]) > 120:
            history["labels"] = history["labels"][-120:]
            history["temperature"] = history["temperature"][-120:]
            history["pm25"] = history["pm25"][-120:]

        final_upload_data["stations_data"][station] = {
            "current": new_current,
            "history": history
        }

    # 寫回 Firebase
    print("📤 正在將累積後的資料存回 Firebase...")
    doc_ref.set(final_upload_data)
    print(f"✅ 真實歷史數據累積成功！目前累積筆數：{len(final_upload_data['stations_data']['斗六']['history']['labels'])} 筆")

if __name__ == '__main__':
    fetch_and_upload()