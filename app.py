from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained CNN model
model = load_model('cnn_model.h5')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMG_SIZE = 128   # change according to your model

@app.route('/', methods=['GET', 'POST'])
def home():

    prediction = None
    image_path = None

    if request.method == 'POST':

        file = request.files['file']

        if file:

            image_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                file.filename
            )

            file.save(image_path)

            # Preprocess image
            img = image.load_img(
                image_path,
                target_size=(IMG_SIZE, IMG_SIZE)
            )

            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            result = model.predict(img_array)

            if np.argmax(result) == 0:
                prediction = "With Mask"
            else:
                prediction = "Without Mask"

    return render_template(
        'index.html',
        prediction=prediction,
        image_path=image_path
    )

if __name__ == '__main__':
    app.run(debug=True)