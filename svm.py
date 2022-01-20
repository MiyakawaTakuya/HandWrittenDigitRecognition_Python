"""
    SVMアルゴリズムで手書き文字の判定を学習し、また結果を評価.
"""
import os
from sklearn import svm,metrics
import joblib

# 学習用データの数
SIZE_TRAINING = 5000

# 検証用データの数
SIZE_TEST = 5000

        
def load_data(type_, size):
    """
        イメージとラベルのデータを取得して返却.
        またここで学習しやすいように、各数値を256で割って1以下の数値にします.
        @param {String} type_ - one of { training | test }
        @param {Int} size - 返却する要素数
    """
   # Load training data. 配列
#    with open("./csv/train-images.csv") as f:
#        images = f.read().split("\n")[:500]
#    with open("./csv/train-labels.csv") as f:
#        labels = f.read().split("\n")[:500]
# %記法とは%演算子を使用して、変数で置き換えてデータを表現する手法。
# 参考URL: https://techacademy.jp/magazine/46444#sec2
    with open(os.path.join("csv", "%s_image.csv" % type_)) as f:
        images = f.read().split("\n")[:size]
    with open(os.path.join("csv", "%s_label.csv" % type_)) as f:
        labels = f.read().split("\n")[:size]

    # Convert data.  配列のままだと処理できないので数字に変換する
    images = [[int(i)/256 for i in image.split(",")] for image in images]
    labels = [int(l) for l in labels]
    
    return images, labels


if __name__ == "__main__":
    
    images, labels = load_data("training", SIZE_TRAINING)
    # Use SVM. 学習処理  clfは分類 クラスフィアー
    print("学習開始")
    clf = svm.SVC()
    clf.fit(images, labels)
    #テストデータを取得
    images, labels = load_data("test", SIZE_TEST)
    
    #予測
    print("予測開始")
    predict = clf.predict(images)
    
    #結果を表示する
    print("結果はこちら")
    ac_score = metrics.accuracy_score(labels, predict)
    print("正解率は", ac_score)
    cl_report = metrics.classification_report(labels, predict)
    print(cl_report)
    
    #結果を保存
    if not os.path.exists("result"):
        os.mkdir("result")
    joblib.dump(clf, os.path.join("result", "svm.pkl"))
    

"""
        `csv`フォルダからデータを読み込み、SVMアルゴリズムを用いた学習を行う。
        そして学習結果を`result`フォルダに`svm.pkl`という名前で保存。
        実装ステップ：
            ・トレーニングデータを読み込む
            ・SVGアルゴリズムによる学習を行う
            ・テストデータを読み込む
            ・精度とメトリクスによる性能評価を行う
            ・学習結果を`result/svm.pkl`ファイルとして保存する
"""
