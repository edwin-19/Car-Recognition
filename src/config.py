img_width, img_height = 224, 224
nb_train_samples = 6500
nb_validation_samples = 1603
epochs = 100
batch_size = 32
n_classes = 196

# Load VGG 16 Model Architecture and Weights
model_path = 'models/vgg16/vgg16_finalModel.json'
model_weights = 'models/vgg16/vgg16_finalModel.h5'

# Load Resnet 50 architecture and weights
# model_weights = 'models/resnet/resnet50.h5'
# model_path = 'models/resnet/resnet50.json'