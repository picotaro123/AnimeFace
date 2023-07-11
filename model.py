import os
import cv2
import numpy as np
import pickle
import random
from tensorflow.keras.preprocessing.image import load_img

def predict(input_filename):
    # モデルの読み込み
    with open("model.pickle", "rb") as file:
        model = pickle.load(file)

    # 認識させたい画像の読み込み
    input_image = cv2.imread(input_filename)
    input_image = cv2.resize(input_image, (64, 64))  # 必要なサイズにリサイズ

    # 画像をフラット化して特徴ベクトルに変換
    input_feature_vector = input_image.flatten()
    input_feature_vector = input_feature_vector.reshape(1, -1)

    # 特徴ベクトルの次元数を調整
    if input_feature_vector.shape[1] > 12288:
        input_feature_vector = input_feature_vector[:, :12288]
    elif input_feature_vector.shape[1] < 12288:
        input_feature_vector = np.pad(
            input_feature_vector, ((0, 0), (0, 12288 - input_feature_vector.shape[1]))
        )

    # モデルの予測
    predicted_label = model.predict(input_feature_vector)

    # 予測されたラベルに該当する画像をデータセットから取得し出力
    image_dir = "data"
    label = predicted_label[0]

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_dir, filename)
            image_label = int(filename.split(".")[0])

            if image_label == label:
                image = cv2.imread(image_path)
                cv2.imshow("Predicted Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return 'success'  # 予測成功を示す値を返す

      # 画像ファイルからランダムに1つを選択
    random_image = random.choice('data')

    # ランダムに選んだ画像ファイルのパスを出力
    random_image_path = os.path.join(random_image)
    print("Random image path:", random_image_path)
    return 'failure'  # 予測失敗を示す値を返す