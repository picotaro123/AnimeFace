import os
import cv2
import numpy as np
import pickle

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

    return predicted_label[0]

input_filename = input('画像のパスを入力してください')  # 認識させたい画像のファイル名
predicted_label = predict(input_filename)
print("Predicted Label:", predicted_label)