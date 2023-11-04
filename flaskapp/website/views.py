from flask import Blueprint, render_template, request,redirect, url_for, flash, Response, send_from_directory
from flask_login import login_required, current_user
from flask import current_app
from tensorflow import keras
import tensorflow as tf
import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename


views = Blueprint('views', __name__)

# Load the model
model = keras.models.load_model('custom_model.h5')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess_image(image_path):
    # Open image
    img = Image.open(image_path)

    # Check if the image shape is (32, 32, 3)
    if img.size == (32, 32) and img.mode == "RGB":
        img = img.convert("RGB")
        print("good img")
    else:
        # Resize and convert to the shape (32, 32, 3)
        img = img.resize((32, 32))
        img = img.convert("RGB")
        img = np.expand_dims(img, axis=0)
        print("bed img")

    # Convert image to numpy array
    img_array = np.array(img)

    return img_array

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    user_choice = 0

    if request.method == "POST":
        user_choice = request.form.get("fname")

        if(user_choice == "1"):

            file = request.files['uimage']

            # try:

            file.save(os.path.join(current_app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
            flash('file save sucessfully', category='success')

            img = preprocess_image(os.path.join(current_app.config['UPLOAD_FOLDER'],file.filename))

            predictions = class_names[np.argmax(model.predict(img))]

            print(predictions)

            os.remove(os.path.join(current_app.config['UPLOAD_FOLDER'],file.filename))

            return f'''
            <html>
                <body>
                    <h1>{predictions}</h1>
                </body>
            </html>
            '''

        

           
        else:
            flash('add the values and then try again', category='error')

    return render_template("home.html", user=current_user)