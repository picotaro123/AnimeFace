import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from model import predict
import random

UPLOAD_FOLDER = './static'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_user_files():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            result = predict(file_path)  # 予測結果を取得

            if result == 'success':
                image = cv2.imread(image_path)
                cv2.imshow("Predicted Image", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return 'success'  # 予測成功を示す値を返す
            else:
                 # 画像ファイルからランダムに1つを選択
                random_image = random.choice('data')

                # ランダムに選んだ画像ファイルのパスを出力
                random_image_path = os.path.join(random_image)
                print("Random image path:", random_image_path)
                return 'failure'  # 予測失敗を示す値を返す

    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)