import requests
import os
import math
import sys
from datetime import datetime
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


# 嘗試引入大腦，如果還是找不到就嘗試加入當前工作目錄
try:
    from service.lstm_predict import predict_pm25
except ModuleNotFoundError:
    sys.path.insert(0, os.getcwd())
    from service.lstm_predict import predict_pm25

load_dotenv(os.path.join('../.env'))
API_KEY = os.getenv("MOENV_API_KEY")
API_URL = f"https://data.moenv.gov.tw/api/v2/aqx_p_145?api_key={API_KEY}&limit=1000&sort=monitordate desc&format=JSON"

DEMO_MODE = True

# --- 2. 初始化 Firebase ---
if not firebase_admin._apps:
    # 確保連線金鑰的路徑也是正確的 (放在根目錄)
    cred_path = os.path.join('../firebase-key.json')
    if not os.path.exists(cred_path):
        print(f"⚠️ 警告：找不到 {cred_path}，請確認金鑰位置。")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
db = firestore.client()

TARGET_STATIONS = ['斗六', '崙背', '臺西', '麥寮']


def fetch_and_upload():
    print(f"🚀 啟動自動化管線 (DemoMode: {DEMO_MODE})")

    response = requests.get(API_URL)
    data = response.json()

    # 🛠️ 徹底修復 'list' object has no attribute 'get' 的報錯
    # 使用 Python 最保險的 try-except 邏輯
    try:
        # 假設它是字典格式，嘗試拿 records
        records = data.get('records', [])
    except AttributeError:
        # 如果噴錯，代表 data 本身就是一個 list
        records = data

    current_data_map = {}
    update_time = ""

    for item in records:
        name = item.get('sitename')
        if name in TARGET_STATIONS:
            if name not in current_data_map:
                current_data_map[name] = {"pm25": 0.0, "temp": 0.0, "rh": 0.0, "ws": 0.0, "wd": 0.0}
                update_time = item.get('monitordate')

            eng = item.get('itemengname')
            val_str = item.get('concentration')
            try:
                if not val_str or str(val_str).strip().upper() in ['X', 'NR', 'ND']:
                    val = 0.0
                else:
                    val = float(val_str)
            except ValueError:
                val = 0.0

            if eng == 'PM2.5':
                current_data_map[name]["pm25"] = val
            elif eng == 'AMB_TEMP':
                current_data_map[name]["temp"] = val
            elif eng == 'RH':
                current_data_map[name]["rh"] = val
            elif eng == 'WS_HR':
                current_data_map[name]["ws"] = val
            elif eng == 'WD_HR':
                current_data_map[name]["wd"] = val

    try:
        dt_obj = datetime.strptime(update_time, "%Y-%m-%d %H:%M:%S")
        short_time = dt_obj.strftime("%m/%d %H:%M")
    except:
        short_time = update_time[5:16].replace("-", "/") if update_time else "未知時間"

    # 讀取 Firebase
    doc_ref = db.collection('air_quality').document('latest')
    doc_snap = doc_ref.get()
    old_data = doc_snap.to_dict() if doc_snap.exists else {"stations_data": {}}

    final_upload_data = {"status": "success", "update_time": update_time, "stations_data": {}}

    for station in TARGET_STATIONS:
        s_data = current_data_map.get(station, {"pm25": 0.0, "temp": 0.0, "rh": 0.0, "ws": 0.0, "wd": 0.0})
        rad = math.radians(s_data["wd"])
        wx = round(s_data["ws"] * math.cos(rad), 4)
        wy = round(s_data["ws"] * math.sin(rad), 2)

        history = old_data.get("stations_data", {}).get(station, {}).get("history", {
            "labels": [], "pm25": [], "temperature": [], "humidity": [], "wind_x": [], "wind_y": []
        })

        # 補齊歷史長度防呆
        current_length = len(history.get("labels", []))
        for key in ["humidity", "wind_x", "wind_y"]:
            if key not in history or len(history[key]) < current_length:
                history[key] = [0.0] * current_length

        if len(history["labels"]) > 0 and history["labels"][-1] == short_time:
            history["temperature"][-1] = s_data["temp"];
            history["pm25"][-1] = s_data["pm25"]
            history["humidity"][-1] = s_data["rh"];
            history["wind_x"][-1] = wx;
            history["wind_y"][-1] = wy
        else:
            history["labels"].append(short_time);
            history["temperature"].append(s_data["temp"])
            history["pm25"].append(s_data["pm25"]);
            history["humidity"].append(s_data["rh"])
            history["wind_x"].append(wx);
            history["wind_y"].append(wy)

        if len(history["labels"]) > 120:
            for key in ["labels", "temperature", "pm25", "humidity", "wind_x", "wind_y"]:
                history[key] = history[key][-120:]

        # 3. 預測
        h_pm25, h_temp, h_rh, h_wx, h_wy = history["pm25"][-24:], history["temperature"][-24:], history["humidity"][
            -24:], history["wind_x"][-24:], history["wind_y"][-24:]

        if len(h_pm25) < 24 and DEMO_MODE:
            needed = 24 - len(h_pm25)
            h_pm25 = [s_data["pm25"]] * needed + h_pm25
            h_temp = [s_data["temp"]] * needed + h_temp
            h_rh = [s_data["rh"]] * needed + h_rh
            h_wx = [wx] * needed + h_wx
            h_wy = [wy] * needed + h_wy

        prediction = None
        if len(h_pm25) == 24:
            prediction = predict_pm25(h_pm25, h_temp, h_rh, h_wx, h_wy, station=station)
            print(f"🔮 {station} 預測完畢: {prediction}")

        final_upload_data["stations_data"][station] = {
            "current": {"pm25": s_data["pm25"], "temperature": s_data["temp"], "humidity": s_data["rh"], "wind_x": wx,
                        "wind_y": wy},
            "history": history, "prediction_lstm": prediction
        }

    doc_ref.set(final_upload_data)
    print("✅ 全自動預測與儲存流程完成")


if __name__ == '__main__':
    fetch_and_upload()