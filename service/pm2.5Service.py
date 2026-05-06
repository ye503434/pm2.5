import os

from flask import Flask, jsonify
from flask_cors import CORS
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)
API_KEY = os.getenv("MOENV_API_KEY")
API_URL =  f'https://data.moenv.gov.tw/api/v2/aqx_p_145?api_key={API_KEY}&limit=1000&sort=monitordate desc&format=JSON'

@app.route('/api/air-quality', methods=['GET'])
def get_air_quality():
    try:

        response = requests.get(API_URL)
        data = response.json()
        print("目前的 API_KEY 是:", API_KEY)
        print("環境部 API 回傳的資料:", data)
        records = data if isinstance(data, list) else data.get('records', [])
        pm25_val = 0
        temp_val = 0
        hum_val = 0
        monitor_date = ""

        for item in records:
            if item.get('sitename') == '斗六':
                monitor_date = item.get('monitordate')
                item_eng = item.get('itemengname')
                val_str = item.get('concentration')

                # 清洗無效數據 (例如 'x')
                try:
                    val = float(val_str) if val_str and val_str.lower() != 'x' else 0
                except ValueError:
                    val = 0

                if item_eng == 'PM2.5':
                    pm25_val = val
                elif item_eng == 'AMB_TEMP':
                    temp_val = val
                elif item_eng == 'RH':
                    hum_val = val

        # 預留 AI 預測的接口 (目前先寫死一個簡單的邏輯做測試)
        # 之後這裡就是放你 PyTorch model 推論的程式碼
        predicted_pm25 = round(pm25_val * 1.1, 1) if pm25_val else 0

        # 打包成乾淨的 JSON 格式回傳給前端
        return jsonify({
            "status": "success",
            "site": "斗六 (Douliu)",
            "update_time": monitor_date,
            "current": {
                "pm25": pm25_val,
                "temperature": temp_val,
                "humidity": hum_val
            },
            "prediction": {
                "pm25_1hr": predicted_pm25
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    # 啟動本地伺服器，使用 5000 port
    app.run(debug=True, port=5000)





