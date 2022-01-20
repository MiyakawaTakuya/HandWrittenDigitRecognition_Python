from os import path
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS, cross_origin
import joblib

app = Flask(__name__)
CORS(app, support_credentials=True)  #CORS対策
# @cross_origin(supports_credentials=True)
# def login():
#   return jsonify({'success': 'ok'})


@app.route("/")
def index():
    """HTMLページをレンダリング."""
    return render_template("index.html")

@app.route("/api/judge")
def api_judge():
    """手書き文字を判定します"""

    # クライアントからのデータを受け取る.
    data = request.args.get("data").split(",")
    data = [int(d) / 256 for d in data]
    print(data)
    
    # 学習結果を読み込み
    pklfile= path.join("result", "svm.pkl")
    clf = joblib.load(pklfile)
    
    # 予測する.
    predict = clf.predict([data])
    print("結果は:", predict)   
    
    return str(predict.tolist()[0])

    """
        学習済みのモデル（SVM）を読み込み、手書き文字が何の数値なのかを判定。
        判定結果をAPIで返すことでフロントエンドに伝える。

        実装ステップ：
            ・学習結果（`result/svm.pkl`）を読み込む
            ・クライアントから渡ってきたデータをもとに、予測を行う
            ・予測結果を返す
    """

if __name__ == "__main__":
    app.run(debug=True, port=5002)
