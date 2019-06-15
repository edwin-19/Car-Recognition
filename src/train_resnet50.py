from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, Activation, add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import log_loss

import os
import urllib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config 

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    The identity_block is the block that has no conv layer at shortcut
    Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    x = Conv2D(nb_filter1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

def download_file(url, path):
    file_name = url.split('/')[-1]
    u = urllib.request.urlopen(url)
    f = open(path + file_name, 'wb')
    meta = u.info()
    file_size = int(meta.get_all("Content-Length")[0])
    print("Downloading: %s Bytes: %s" % (file_name, file_size))
    
    file_size_dl = 0
    block_sz = 8192
    
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
        
        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print(status)
    
    f.close()

def resnet50_model(img_rows, img_cols, color_type=1, num_classes=None):
    """
    Resnet 50 Model for Keras
    Model Schema is based on 
    https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
    ImageNet Pretrained Weights 
    https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of class labels for our classification task
    """
    '''
    ValueError: Dimension 2 in both shapes must be equal, but are 3 and 64. 
    Shapes are [7,7,3,64] and [7,7,64,3]. for 'Assign_1' (op: 'Assign') with input shapes: [7,7,3,64], [7,7,64,3].
    '''
    # Handle Dimension Ordering for different backends
    global bn_axis
    
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type))
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols))
    
    # Block 1
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # Block 2
    x = conv_block(x, 3, (64, 64, 256), stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x , 3, [64, 64, 256], stage=2, block='c')
    
    # Block 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x , 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x , 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x , 3, [128, 128, 512], stage=3, block='d')
    
    # Block 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    
    # Block 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x , 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    
    # Fully Connected Softmax Layer
    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)
    
    # Create model
    model = Model(img_input, x_fc)
    
    # Load ImageNet pre-trained data 
    if K.image_dim_ordering() == 'th':
        # Use pre-trained weights for Theano backend
        weights_path = 'models/resnet/resnet50_weights_th_dim_ordering_th_kernels.h5'
    else:
        # Use pre-trained weights for Tensorflow backend
        weights_path = 'models/resnet/resnet50_weights_th_dim_ordering_th_kernels.h5'

    model.load_weights(weights_path, reshape=True)
    
    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc10')(x_newfc)
    
    # Create another model with our customized softmax
    model = Model(img_input, x_newfc)
    
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def save_model(model, fileName):
    model_json = model.to_json()
    with open(fileName + ".json", "w") as json_file:
        json_file.write(model_json)
    
    model.save_weights(fileName + ".h5")
    print("Saved model to disk")

def show_accuracy(history, filename):
    plt.plot(history.history['val_acc'], 'r')
    plt.plot(history.history['acc'], 'b')
    plt.title('Performance of model ' + 'VGG16')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs No')
    plt.show()

if __name__ == "__main__":
    train_path = 'data/train/'
    test_path = 'data/test/'

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
        target_size=(config.img_width, config.img_height),
        batch_size=config.batch_size,
        class_mode='categorical')


    validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(config.img_width, config.img_height),
        batch_size=config.batch_size,
        class_mode='categorical')

    weights_path = 'models/resnet/resnet50_weights_th_dim_ordering_th_kernels.h5'

    if not os.path.exists('models/resnet/'):
        os.makedirs('models/resnet/')

    if not os.path.exists(weights_path):
        url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5'
        download_file(url, 'models/resnet/')

    model = resnet50_model(config.img_height, config.img_width, 3, config.n_classes)

    print(model.summary())

    checkpoint = ModelCheckpoint('models/resnet/resnet50.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='max')
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, mode='auto')
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=config.nb_train_samples // config.batch_size, 
        epochs=config.epochs,
        validation_data=validation_generator, 
        validation_steps=config.nb_validation_samples // config.batch_size,
        callbacks=[earlystop, checkpoint]
    )

    show_accuracy(history, 'models/resnet/resnet50' + '_initialModel_plot.png')
    save_model(model, 'models/resnet/resnet50')