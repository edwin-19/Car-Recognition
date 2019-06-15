# Car Image Classfier 
The following repo is an image classfier for cars built for the grab AI for SEA Challenge (Computer Vision Challenge)
<!-- 
## Notes
- Uses the standford car dataset https://ai.stanford.edu/~jkrause/cars/car_dataset.html
- Used two architecture -->

## Enviroment
- Ubuntu 18.04
- Anaconda (Python 3.6.8)
- Keras (Tensorflow Backend)
- Nvidia-GTX 1080

## Installation
```sh
git clone https://github.com/edwin-19/Car-Recognition.git
pip install -r requirements.txt
```

## Evaluation
<b>NOTE: Change the config file for model path</b>  
To run the script make sure to download the architecture and weights from the following [Google Drive](https://drive.google.com/drive/folders/10PjimksZGUnPSdXDO6eIE5ui2qcwwV1e?usp=sharing) and add it to models folder e.g: 
- models/vgg16
- models/resnet
### Single Predict - from single image
```sh
python predict.py --image dataset/cars_test/00001.jpg --show_image
```

### Batch Predict - from folder
```sh
python batch_predict.py --image_folder dataset/test/
```

<b>NOTES: Results are classfied_name followd by - then the perecentage:  
Suzuki Aerio Sedan 2007 - 58.42927694320679.jpg
</b>

### Python API
Using FLASK to serve as a flask rest api:
```sh
python app.py
```
Use POSTMAN or any other client to parse the following link() and set it to post as such:
![POSTMAN](https://github.com/edwin-19/Car-Recognition/blob/master/src/images/Postman.png?raw=true, "POSTMAN")

## Training
### Download Dataset
The dataset used to train the classfier is the [Standford Car Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) and put inside the raw_data folder as such: 

Download the following three:
- Training Image
- Testing Image 
- Car Devkit  

![file_structure](https://github.com/edwin-19/Car-Recognition/blob/master/src/images/FileStrucutre.png?raw=true)  

then unzip all tar files and add to the following directory inside src:


Or you can just run the following script to download and unzip:
```sh
python download_cardataset.py
```

### Generate Annotation
Generate labels from mat file from car devkit and convert them into a CSV format:
```sh
python generate_annotation.py
```

### Split dataset by 8:2 ratio and label pictures
Split the dataset by 8:2 ratio as the test dataset has no labels so we will split from training set and move the training images into files labelled
```sh
python split_train_test.py
```

### Run training script
<b>NOTE: For this repo VGG16 is the most recommended architecture and Resnet50 was included for the sake of trying out certain architectures but accuracy is lesser than VGG16</b>  
For this repo i have used tranfer learning from pretrained weights using the following architectures:
- VGG16
- It trains twice, the second time to fine tune the weights
    - ```sh
        python train_vgg16.py
        ```
- Resnet50
- The pretrained weights were downloaded from another repo [here](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5) (script will automatically download if there isn't any)
    - ```sh
        python train_resnet50.py
        ``` 
### Evaluation Script
Run evaluation script to print report and confusion matrix
```sh
python evaluate.py
```

- Model architecture used is vgg16 
# Model accuracy and Loss
- Tested on training set with an accuracy of 70+%
## Accuracy
![Alt text](https://github.com/edwin-19/Car-Recognition/blob/master/src/models/vgg16/VGG16_Final_acc.png?raw=true "Accuracy")

## Loss
![Alt text](https://github.com/edwin-19/Car-Recognition/blob/master/src/models/vgg16/VGG16_Final_loss.png?raw=true "Accuracy")

# CREDIT
The following code was mostly insipired based on a similar challenge by KAGGLE: 
https://www.kaggle.com/jutrera/training-a-densenet-for-the-stanford-car-dataset  
<b>I Borrowed the evaluation method from the following notebook - I DO NOT take credit for it</b>