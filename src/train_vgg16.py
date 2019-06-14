from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, regularizers
from keras import backend as K

from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping

from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from matplotlib import pyplot as plt
import os
import config

train_path = 'data/train/'
test_path = 'data/test/'

def build_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(config.img_width, config.img_height, 3))
    for layer in base_model.layers:
        layer.trainable = False
    
    # Flatten the results from conv block
    x = Flatten()(base_model.output)
    
    #add another fully connected layers with batch norm and dropout
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.8)(x)
    
    #add another fully connected layers with batch norm and dropout
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.8)(x)
    
    #add logistic layer with all car classes or output layer
    output = Dense(
        config.n_classes, activation='softmax', 
        kernel_initializer='random_uniform', bias_initializer='random_uniform', 
        bias_regularizer=regularizers.l2(0.01), name='output'
    )(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    return model

def save_model(model, fileName):
    model_json = model.to_json()
    with open(fileName + ".json", "w") as json_file:
        json_file.write(model_json)
    
    model.save_weights(fileName + ".h5")
    print("Saved model to disk")
    
def plot_learning_curves(save_path, model_name, optimizer, history, show_plots):
    # Display and save learning curves.
    
    fig=plt.figure(figsize=(8, 8))
    fig.add_subplot(2, 2, 2)
    
    # accuracy 
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy of the model')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['validation set','training set'], loc='lower right')
    plt.savefig(save_path + "/" + model_name + '_acc.png')
    
    # loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss of the model')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['validation set','training set'], loc='upper right')
    plt.savefig(save_path + "/" + model_name + '_loss.png')
    
    if show_plots:
        plt.show()

def augment_data(img_width, img_height, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        #shear_range=0.2,
        zoom_range=0.2,
        #fill_mode = 'constant',
        #cval = 1,
        rotation_range = 5,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')


    validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    
    return train_generator, validation_generator

if __name__ == "__main__":
    if not os.path.exists('models/'):
        os.makedirs('models/')
        if not os.path.exists('models/vgg16'):
            os.makedirs('models/vgg16')
    
    train_generator, validation_generator = augment_data(config.img_width, config.img_height, config.batch_size)
    
    base_model = build_vgg16_model()

    # Set initial training 
    rmsprop = RMSprop(decay=1e-4, lr=0.0001)
    base_model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    print(base_model.summary())
    
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')
    history = base_model.fit_generator(
        train_generator,
        steps_per_epoch=config.nb_train_samples // config.batch_size, 
        epochs=config.epochs,
        validation_data=validation_generator, 
        validation_steps=config.nb_validation_samples // config.batch_size,
        callbacks=[earlystop]
    )

    plot_learning_curves('models/vgg16/', 'VGG16_Initial', 'rmsprop', history, False)
    save_model(base_model, 'models/vgg16/vgg16_initialModel')

    # Set layer to retrain
    for layer in base_model.layers[:15]:
        layer.trainable = False

    for layer in base_model.layers[15:]:
        layer.trainable = True
    
    # Fine Tune Training
    sgd = SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
    # sgd = Adam(lr=0.0001, decay=1e-4)
    base_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')

    history = base_model.fit_generator(
        train_generator,
        steps_per_epoch=config.nb_train_samples // config.batch_size, 
        epochs=config.epochs,
        validation_data=validation_generator, 
        validation_steps=config.nb_validation_samples // config.batch_size,
        callbacks=[earlystop]
    )
    
    plot_learning_curves('models/vgg16/', 'VGG16_Final', 'sgd', history, False)
    save_model(base_model, 'models/vgg16/vgg16_finalModel')