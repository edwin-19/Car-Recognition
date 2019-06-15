import numpy as np 
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import glob

df = pd.read_csv('dataset/car_train.csv')
test_df = pd.read_csv('dataset/car_test.csv')

train_path = 'data/train/'
test_path = 'data/test/'

train_image_path = 'dataset/cars_train/'
test_image_path = 'dataset/cars_test/'

img_width, img_height = 224, 224

# Create folders for all class
if not os.path.exists('data/'):
    os.makedirs('data/')

if not os.path.exists(train_path):
    os.makedirs(train_path)

if not os.path.exists(test_path):
    os.makedirs(test_path)

for c in df['class'].unique():
    if not os.path.exists(train_path + c):
        os.mkdir(train_path + c.replace('/', '\\'))
        os.mkdir(test_path + c.replace('/', '\\'))

# Split dataframe to train test 8:2 ratio
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

for index, row in train.iterrows():
    if os.path.exists(train_image_path + row['fname']):
        if not os.path.exists(train_path + row['fname']):
            src_img = cv2.imread(train_image_path + row['fname'])
            (x1, y1, x2, y2) = row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']
            height, width = src_img.shape[:2]
            
            # margins of 16 pixels
            margin = 16
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(x2 + margin, width)
            y2 = min(y2 + margin, height)
    
            crop_image = src_img[y1:y2, x1:x2]
    
            dst_img = cv2.resize(crop_image, (img_height, img_width))
            dst_path = train_path + row['class'] + '/' + row['fname']
            
            cv2.imwrite(dst_path, dst_img)

for index, row in test.iterrows():
    if os.path.exists(train_image_path + row['fname']):
        if not os.path.exists(test_path + row['fname']):
            src_img = cv2.imread(train_image_path + row['fname'])
            (x1, y1, x2, y2) = row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']
            height, width = src_img.shape[:2]
            
            # margins of 16 pixels
            margin = 16
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(x2 + margin, width)
            y2 = min(y2 + margin, height)
    
            crop_image = src_img[y1:y2, x1:x2]
    
            dst_img = cv2.resize(crop_image, (img_height, img_width))
            dst_path = test_path + row['class'] + '/' + row['fname']
            
            cv2.imwrite(dst_path, dst_img)