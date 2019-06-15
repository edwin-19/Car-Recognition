from flask import Flask, request, make_response, request, abort, redirect, send_file, jsonify

from keras.models import model_from_json
from keras.optimizers import SGD
import keras.backend as K 

import numpy as np 
import cv2 
import pandas as pd 
import config 
import logging

app = Flask(__name__)

# Load all class names first
class_df = pd.read_csv('dataset/classess.csv')

def load_model(model_path, model_weights):
    with open(model_path, 'r') as f:
        model = model_from_json(f.read())
    
    model.load_weights(model_weights)
    sgd = SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Left as such might want to add an index page but will consider later
@app.route('/')
def index():
    return 'Index'

@app.route('/recognize_car', methods=['POST'])
def upload():
    try:
        # loading model
        model = load_model(config.model_path, config.model_weights)

        # Get file from request and convert to uint8 format
        image = request.files['image'].read()
        image_array = np.fromstring(image, np.uint8)
        unchanged_image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        img = unchanged_image.copy()
        
        # normalize img to float32
        img = cv2.resize(img, (config.img_width , config.img_height))
        img = img / 255.
        img = np.expand_dims(img, axis=0)
        
        # Get predictions from model
        prediction = model.predict(img)
    
        class_pred = np.argmax(prediction)
        percentage = np.max(prediction)
        
        # Clear model once done loading
        K.clear_session()
        
        predicted_class = class_df['class'][class_pred]
        
        data = {
            'predicted_class': predicted_class,
            'percentage': str(percentage * 100)
        }

        # Return dictionary data as json
        return make_response(jsonify(data))
    except Exception as err:
        logging.error('An error has occurred whilst processing the file: "{0}"'.format(err))
        abort(400)

# Error handling
@app.errorhandler(400)
def bad_request(erro):
    return make_response(jsonify({'error': 'We cannot process the file sent in the request.'}), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Resource no found.'}), 404)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)