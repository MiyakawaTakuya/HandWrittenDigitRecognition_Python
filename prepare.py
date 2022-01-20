"""
    SVM適用前のデータの前処理.
    MNISTファイル(gzip)を、CSVファイルに変換.
"""
import os
import struct
import gzip


#①バイナリデータをstructモジュールで扱う
#サンプルコード
#f = open(“file.binary", "rb")  ファイル読み込みの関数  rbでバイナリデータとして読み込んでの指示
#num1, num2 = struct.unpack(">II", f.read(8))  8バイト分のデータを読み込んで、IIで解釈して
#structnの中の定義で"I"はunsigned int integer 4byte を指す
#">"は読み取り方法

def csv_label(fname, type_):
    """
        ラベルデータを出力.
        @param {String} fname - MNISTのファイル名
        @param {String} type_ - one of { training | test }
    """
    print("%s processing..." % fname)

    # ラベルデータをGzipファイルから読み取ります.
    with gzip.open(os.path.join("mnist", fname), "rb") as f:
        _, cnt = struct.unpack(">II", f.read(8))
        labels = []
        for i in range(cnt):
            label = str(struct.unpack("B", f.read(1))[0])
            labels.append(label)

    # CSV結果として出力します.
    with open(os.path.join("csv", type_ + "_label.csv"), "w") as f:
        f.write("\n".join(labels))
        
# Read MNIST `label`.
# fpath = "./mnist/train-labels-idx1-ubyte.gz"
# with gzip.open(fpath, "rb") as f:  #gzip.openで解凍することなく読み込むことができる
#     magic_number, img_count = struct.unpack(">II", f.read(8))
#     labels = []
#     for i in range(img_count):  #60000件
#         label = str(struct.unpack("B", f.read(1))[0])  #1byte分をunsigned byte"B"のかたで読み込む　タプルの[０]番目
#         labels.append(label)

# # Write as csv.
# outpath = './csv/train-labels.csv'
# with open(outpath, "w") as f:  #"w"書き出しモード
#     f.write("\n".join(labels))  #行で処理
    


#②画像データを扱う
def csv_image(fname, type_):
    """
        画像データを出力します.
        @param {String} fname - MNISTのファイル名
        @param {String} type_ - one of { training | test }
    """
    print("%s processing..." % fname)

    # 画像データをGzipファイルから読み取ります.
    with gzip.open(os.path.join("mnist", fname), "rb") as f:
        _, cnt, rows, cols = struct.unpack(">IIII", f.read(16))
        # 画像読み込み
        images = []
        for i in range(cnt):
            binarys = f.read(rows * cols)
            images.append(",".join([str(b) for b in binarys]))

    # CSV結果として出力します.
    with open(os.path.join("csv", type_ + "_image.csv"), "w") as f:
        f.write("\n".join(images))

# Read MNIST `images`.
# fpath = "./mnist/train-images-idx3-ubyte.gz"
# with gzip.open(fpath, "rb") as f:
#     _, img_count = struct.unpack(">II", f.read(8))  #最初の8byte分を読み込む 不要なものは_
#     rows, cols = struct.unpack(">II", f.read(8))  #次の8byte分を読み込む ピクセルの行列単位
#     images = []
#     for i in range(img_count):
#         binary = f.read(rows * cols)
#         images.append(",".join([str(b) for b in binary])) #一次変数bに1byteの0〜255の数字を１画像分の文字列にしてる

if __name__ == "__main__":

    if not os.path.exists("csv"):
        os.mkdir("csv")

    # トレーニングデータ.
    csv_image("train-images-idx3-ubyte.gz", "training")
    csv_label("train-labels-idx1-ubyte.gz", "training")

    # テストデータ.
    csv_image("t10k-images-idx3-ubyte.gz", "test")
    csv_label("t10k-labels-idx1-ubyte.gz", "test")

    """
        **** ここを実装します（基礎課題） ****
        `mnist`フォルダにあるデータから、CSVを作成し、`csv`フォルダに出力するプログラムを作成してください。
        実装方法は、講義資料や答えを参照してください。
        最初の課題から難易度高めですが、ぜひチャレンジしてみてください！

        作成が完了したら、同ディレクトリにある`check_image.py`を実行し、
        画像が正しく出力されるかを確認してください。
    """