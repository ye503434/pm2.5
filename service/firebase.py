import requests
import os
import math
import sys
from datetime import datetime
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# 設定區：強大路徑定位
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

DEMO_MODE = True

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # 🟢 同時 import LSTM 與 STGNN 的預測函式
    from lstm_predict import predict_pm25
    from stgnn_predict import predict_stgnn
except (ImportError, ModuleNotFoundError):
    sys.path.insert(0, root_dir)
    from service.lstm_predict import predict_pm25
    from service.stgnn_predict import predict_stgnn

load_dotenv(os.path.join(root_dir, '.env'))
API_KEY = os.getenv("MOENV_API_KEY")
API_URL = f"https://data.moenv.gov.tw/api/v2/aqx_p_145?api_key={API_KEY}&limit=1000&sort=monitordate desc&format=JSON"

# 初始化 Firebase
if not firebase_admin._apps:
    cred_path = os.path.join(root_dir, 'firebase-key.json')

    if not os.path.exists(cred_path):
        print(f"❌ 錯誤：找不到金鑰檔！搜尋路徑為: {os.path.abspath(cred_path)}")
        raise FileNotFoundError(f"找不到 firebase-key.json")

    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()
TARGET_STATIONS = ['斗六', '崙背', '臺西', '麥寮']


def fetch_and_upload():
    print(f"🚀 啟動自動化管線 (DemoMode: {DEMO_MODE})")

    try:
        response = requests.get(API_URL, timeout=30)
        data = response.json()
    except Exception as e:
        print(f"❌ API 請求失敗: {e}")
        return

    try:
        records = data.get('records', [])
    except AttributeError:
        records = data

    if not records:
        print("⚠️ 警告：API 未回傳任何資料，結束執行。")
        return

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

    if not update_time:
        print("⚠️ 警告：找不到目標站點的更新時間。")
        return

    try:
        dt_obj = datetime.strptime(update_time, "%Y-%m-%d %H:%M:%S")
        short_time = dt_obj.strftime("%m/%d %H:%M")
    except:
        short_time = update_time[5:16].replace("-", "/")

    doc_ref = db.collection('air_quality').document('latest')
    doc_snap = doc_ref.get()
    old_data = doc_snap.to_dict() if doc_snap.exists else {"stations_data": {}}

    final_upload_data = {"status": "success", "update_time": update_time, "stations_data": {}}

    #準備一個字典，用來一次性收集 4 個站點的資料餵給 STGNN
    stgnn_recent_data = {}

    for station in TARGET_STATIONS:
        s_data = current_data_map.get(station, {"pm25": 0.0, "temp": 0.0, "rh": 0.0, "ws": 0.0, "wd": 0.0})

        rad = math.radians(s_data["wd"])
        wx = round(s_data["ws"] * math.cos(rad), 4)
        wy = round(s_data["ws"] * math.sin(rad), 4)

        # 從舊資料中提取「上一小時所做出的預測值」
        old_station_box = old_data.get("stations_data", {}).get(station, {})
        old_lstm_pred = old_station_box.get("prediction_lstm", None)
        old_stgnn_pred = old_station_box.get("prediction_stgnn", None)

        # 如果先前無預測值（例如首度運行），則安全預設為 0.0 或當前真實值
        old_lstm_val = old_lstm_pred if old_lstm_pred is not None else 0.0
        old_stgnn_val = old_stgnn_pred if old_stgnn_pred is not None else 0.0

        # 在 history 初始化字典中新增 'lstm' 與 'stgnn'
        history = old_data.get("stations_data", {}).get(station, {}).get("history", {
            "labels": [], "pm25": [], "temperature": [], "humidity": [], "wind_x": [], "wind_y": [],
            "lstm": [], "stgnn": []
        })

        # 補齊歷史長度防呆，將新加入的 lstm 與 stgnn 納入自動對齊
        current_length = len(history.get("labels", []))
        for key in ["humidity", "wind_x", "wind_y", "lstm", "stgnn"]:
            if key not in history or len(history[key]) < current_length:
                history[key] = [0.0] * current_length

        if len(history["labels"]) > 0 and history["labels"][-1] == short_time:
            history["temperature"][-1] = s_data["temp"]
            history["pm25"][-1] = s_data["pm25"]
            history["humidity"][-1] = s_data["rh"]
            history["wind_x"][-1] = wx
            history["wind_y"][-1] = wy
            # 如果是同個小時重複執行，維持覆蓋上一小時預言的對齊值
            history["lstm"][-1] = old_lstm_val
            history["stgnn"][-1] = old_stgnn_val
        else:
            history["labels"].append(short_time)
            history["temperature"].append(s_data["temp"])
            history["pm25"].append(s_data["pm25"])
            history["humidity"].append(s_data["rh"])
            history["wind_x"].append(wx)
            history["wind_y"].append(wy)
            # 新的一小時，將上一小時對當前的預測值正式歸檔進歷史陣列
            history["lstm"].append(old_lstm_val)
            history["stgnn"].append(old_stgnn_val)

        # 將 'lstm' 與 'stgnn' 加入 120 筆上限截斷清單中
        if len(history["labels"]) > 120:
            for key in ["labels", "pm25", "temperature", "humidity", "wind_x", "wind_y", "lstm", "stgnn"]:
                history[key] = history[key][-120:]

        h_pm25 = history["pm25"][-24:]
        h_temp = history["temperature"][-24:]
        h_rh = history["humidity"][-24:]
        h_wx = history["wind_x"][-24:]
        h_wy = history["wind_y"][-24:]

        if len(h_pm25) < 24 and DEMO_MODE:
            needed = 24 - len(h_pm25)
            h_pm25 = [s_data["pm25"]] * needed + h_pm25
            h_temp = [s_data["temp"]] * needed + h_temp
            h_rh = [s_data["rh"]] * needed + h_rh
            h_wx = [wx] * needed + h_wx
            h_wy = [wy] * needed + h_wy

        # 將整理好的陣列存入 stgnn_recent_data 供後續 STGNN 使用
        stgnn_recent_data[station] = {
            "pm25": h_pm25,
            "temp": h_temp,
            "rh": h_rh,
            "wind_x": h_wx,
            "wind_y": h_wy
        }

        # 預測當前站點的 LSTM (這代表對未來「下一小時」的全新預測)
        prediction_lstm = None
        if len(h_pm25) == 24:
            prediction_lstm = predict_pm25(h_pm25, h_temp, h_rh, h_wx, h_wy, station=station)
            print(f"🔮 {station} LSTM 預測完畢: {prediction_lstm}")

        final_upload_data["stations_data"][station] = {
            "current": {"pm25": s_data["pm25"], "temperature": s_data["temp"], "humidity": s_data["rh"], "wind_x": wx,
                        "wind_y": wy},
            "history": history,
            "prediction_lstm": prediction_lstm
            # STGNN 預測值稍後補上
        }

    # 迴圈結束，此時我們已經收集了 4 個站的資料，開始進行 STGNN 全局預測
    all_ready_for_stgnn = all(len(stgnn_recent_data[s]["pm25"]) == 24 for s in TARGET_STATIONS)

    if all_ready_for_stgnn:
        try:
            stgnn_predictions = predict_stgnn(stgnn_recent_data)
            print(f"🌐 STGNN 全局預測完畢: {stgnn_predictions}")

            # 將全新的 STGNN 預測結果塞回 Firebase 上傳字典中（代表對未來下一小時的預報）
            for station in TARGET_STATIONS:
                final_upload_data["stations_data"][station]["prediction_stgnn"] = stgnn_predictions.get(station)

        except Exception as e:
            print(f"❌ STGNN 預測發生錯誤: {e}")
            for station in TARGET_STATIONS:
                final_upload_data["stations_data"][station]["prediction_stgnn"] = None
    else:
        print("⚠️ 歷史資料筆數不足，跳過 STGNN 預測")
        for station in TARGET_STATIONS:
            final_upload_data["stations_data"][station]["prediction_stgnn"] = None

    # 最後統一上傳到 Firebase
    doc_ref.set(final_upload_data)
    print("✅ 全自動預測與儲存流程完成")


if __name__ == '__main__':
    fetch_and_upload()