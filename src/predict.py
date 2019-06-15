from keras.models import model_from_json
from keras.optimizers import SGD

import numpy as np
import cv2 
import pandas as pd 
import config 
import argparse
import os

def load_model(model_path, model_weights):
    with open(model_path, 'r') as f:
        model = model_from_json(f.read())
    
    model.load_weights(model_weights)
    sgd = SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
    
if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        '-i','--image', default='dataset/cars_test/00001.jpg'
    )
    argparse.add_argument(
        '-s', '--show_image', action='store_true'
    )

    args = argparse.parse_args()
    
    if not os.path.exists('results'):
        os.mkdir('results')
    
    # Load classes
    class_df = pd.read_csv('dataset/classess.csv')
    # Load Model
    model = load_model(config.model_path, config.model_weights)
    
    # Normalize image
    img = cv2.imread('dataset/cars_test/00001.jpg')
    img = cv2.resize(img, (config.img_width , config.img_height))
    img_copy = img.copy()
    img = img / 255.
    img = np.expand_dims(img, axis=0)
    
    # Predict image
    prediction = model.predict(img)
    
    # Get highest probability
    class_pred = np.argmax(prediction)
    percentage = np.max(prediction)
    
    if args.show_image:
        cv2.imshow(class_df['class'][class_pred] + ' - ' + str(percentage * 100), img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Write to folder image
    cv2.imwrite('results/' + class_df['class'][class_pred] + ' - ' + str(percentage * 100) + '.jpg', img_copy)
    
