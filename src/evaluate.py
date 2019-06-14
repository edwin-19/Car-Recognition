from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd

img_width, img_height = 224, 224
batch_size = 32
n_classes = 196
test_path = 'data/test/'

test_image_dataset = 'dataset/cars_test/'

df = pd.read_csv('dataset/car_train.csv')
class_df = pd.read_csv('dataset/classess.csv')

if __name__ == "__main__":
    # Load Model
    with open('models/vgg16/vgg16_finalModel.json', 'r') as f:
        model = model_from_json(f.read())

    model.load_weights('models/vgg16/vgg16_finalModel.h5')

    sgd = SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
    # sgd = Adam(lr=0.0001, decay=1e-4)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    evaluation = model.evaluate_generator(validation_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False)

    print(evaluation)

    pred = model.predict_generator(validation_generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
    predicted = np.argmax(pred, axis=1)

    class_df = pd.read_csv('dataset/classess.csv')

    print('Confusion Matrix')
    cm = confusion_matrix(validation_generator.classes, np.argmax(pred, axis=1))
    plt.figure(figsize = (30,20))
    sn.set(font_scale=1.4) #for label size
    sn.heatmap(cm, annot=True, annot_kws={"size": 12}) # font size
    plt.show()
    print()
    print('Classification Report')
    print(classification_report(validation_generator.classes, predicted, target_names=df['class'].unique(), labels=range(len(df['class'].unique()))))
    