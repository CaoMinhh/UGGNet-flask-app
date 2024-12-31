from flask import Flask, render_template, request, flash
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt





app = Flask(__name__)

model_path = r"\model\UGG_19train_7layer.keras"

UGG_19 = load_model(model_path)


def predict_image(image_data):
    """
    Dự đoán lớp và vùng bất thường từ ảnh đầu vào.

    Args:
        image_data (numpy.ndarray): Ảnh đầu vào.

    Returns:
        tuple: Bao gồm ảnh đã xử lý, vùng bất thường và nhãn dự đoán.
    """
    gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)  
    img_data_resized = cv2.resize(gray_image, (256, 256))     
    img_array = np.expand_dims(img_data_resized, axis=0)  
    prediction = UGG_19.predict(img_array)


    mapping_dict = {'Malignant - Ác tính': 0, 'Normal - Bình thường': 1, 'Benign - Lành tính': 2}
    y_pred_classes = np.argmax(prediction, axis=1)
    y_pred_label = [key for key, value in mapping_dict.items() if value == y_pred_classes[0]][0]

    prediction_model_1 = (UGG_19.get_layer('functional_1').predict(img_array) > 0.9).astype(np.float32)

    return img_array[0], prediction_model_1[0, :, :, 0], y_pred_label





@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Xử lý yêu cầu từ giao diện chính.

    Returns:
        render_template: Trả về trang HTML kết quả hoặc trang chính.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return render_template('index.html')

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return render_template('index.html')

        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        img, prediction, label = predict_image(image)
        probability = np.max(prediction, axis=1)[0]  
        result_image_path = 'static/result_image.png'
        plt.figure(figsize=(20, 10))
        plt.clf()
        plt.subplot(131)
        plt.imshow(img, cmap='gray')
        plt.title(f'Ảnh siêu âm' , fontsize=25)

        plt.subplot(132)
        plt.imshow(prediction, cmap='gray')
        plt.title(f'Dự đoán vùng bất thường' , fontsize=25)

        plt.subplot(133)
        plt.imshow(img, cmap='gray')
        plt.imshow(prediction, alpha=0.5)
        plt.title(f'Tổn thương: {label} \n Xác suất: ({probability:.2%})' , fontsize=25)
        plt.tight_layout()
        
        plt.savefig(result_image_path)

        return render_template('result.html', img_path=result_image_path, prediction=prediction, label=label)

    return render_template('index.html')

if __name__ == '__main__':
    # app.run(debug=True)
    app.run()