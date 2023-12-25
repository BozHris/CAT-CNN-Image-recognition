from flask import Flask, render_template, redirect, url_for, request, send_from_directory
import os

#the cnn model

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import scipy
import numpy as np
from keras.preprocessing import image



train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
        r'dataset\dataset\training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        r'dataset\dataset\test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))

cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

cnn.fit(x=training_set,validation_data=test_set,epochs=25)



app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
def home():
    return render_template('index.html')





@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # Handle the file here (e.g., save it to the server or process it)
    # For example, you can save the file to the 'uploads' folder:
    file.save('uploads/' + file.filename)
    #test_image = image.load_img(f'uploads/{file.filename}', target_size=(64, 64))

    else:
        test_image = image.load_img(f"uploads/{file.filename}", target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0, )
        result = cnn.predict(test_image)

        if result[0][0]==0:
            prediction='Cat'
            return render_template('index-cat.html')
        else:
            prediction='Something else'
            return render_template('index-no-cat.html')





if __name__ == '__main__':
    app.run(debug=True)