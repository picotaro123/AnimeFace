import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time
import pickle

# 画像が保存されているディレクトリパス
image_dir = "data"

# 画像データとラベルを格納するリスト
images = []
labels = []

# ディレクトリ内の画像ファイルを読み込む
image_files = [filename for filename in os.listdir(image_dir) if filename.endswith(".jpg")]

# 進捗バーを表示しながら画像ファイルを処理
total_images = len(image_files)
progress = 0
for filename in image_files:
    image_path = os.path.join(image_dir, filename)
    label = int(filename.split(".")[0])  # 画像のラベルをファイル名から取得

    # 画像を読み込み、リサイズして特徴ベクトルに変換
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # 必要なサイズにリサイズするなどの前処理を追加
    feature_vector = image.flatten()  # 画像を1次元ベクトルに変換

    images.append(feature_vector)
    labels.append(label)

    # 進捗表示を更新
    progress += 1
    progress_percent = (progress / total_images) * 100
    print(f"Processing images: {progress}/{total_images} ({progress_percent:.2f}%)", end="\r")

# データセットをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# パイプラインを作成してモデルの構築とトレーニング
model = make_pipeline(StandardScaler(), SGDRegressor(loss='squared_error', random_state=42))
iterations = 100  # イテレーション数

# 学習開始のメッセージを表示
print("\nTraining started...")

# 学習の進捗表示を追加
for i in range(iterations):
    model.fit(X_train, y_train)

    # 進捗表示を更新
    progress_percent = ((i + 1) / iterations) * 100
    print(f"Training progress: {i + 1}/{iterations} ({progress_percent:.2f}%)", end="\r")

    time.sleep(0.1)  # 処理が速すぎると進捗表示が見えない場合があるため、適度な待機時間を追加しています

# 進捗表示をクリアして学習完了のメッセージを表示
print("\033[KTraining completed.")

# テストデータでの予測
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# モデルを保存
with open("model.pickle", "wb") as file:
    pickle.dump(model, file)