
import os
import random
# 画像ファイルからランダムに1つを選択
random_image = random.choice('data')

 # ランダムに選んだ画像ファイルのパスを出力
random_image_path = os.path.join(random_image)
print("Random image path:", random_image_path)
