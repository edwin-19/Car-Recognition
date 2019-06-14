import scipy.io as spio
import numpy as np
import pandas as pd

train_mat = spio.loadmat('dataset/devkit/cars_train_annos.mat')
train_mat = {k:v for k, v in train_mat.items() if k[0] != '_'}

# Test mat file 
test_mat = spio.loadmat('dataset/devkit/cars_test_annos.mat')
test_mat = {k:v for k, v in test_mat.items() if k[0] != '_'}

class_mat = spio.loadmat('dataset/devkit/cars_meta.mat')
class_mat = {k:v for k, v in class_mat.items() if k[0] != '_'}

classess = {'class': []}
for k, v in class_mat.items():
    for annot in np.transpose(v[0]):
        for a in annot:
            classess['class'].append(a)

df_class = pd.DataFrame(classess)
df_class.to_csv('dataset/classess.csv', index=False)

bbox_x1 = []
bbox_x2 = []
bbox_y1 = []
bbox_y2 = []
temp_class_name = []
fname = []

for k, v in train_mat.items():
    for annot in np.transpose(v[0]):
        for i, a in enumerate(annot):
            if i == 0:
                bbox_x1.append(np.asscalar(a))
            elif i == 1:
                bbox_y1.append(np.asscalar(a))
            elif i == 2:
                bbox_x2.append(np.asscalar(a))
            elif i == 3:
                bbox_y2.append(np.asscalar(a))
            elif i == 4:
                temp_class_name.append(np.asscalar(a))
            elif i == 5:
                fname.append(np.asscalar(a))
                
class_name = []
for c_name in temp_class_name:
    class_name.append(np.asscalar(class_mat['class_names'][0][c_name - 1]))

train_df = pd.DataFrame(
    {
        'bbox_x1': bbox_x1,
        'bbox_x2': bbox_x2,
        'bbox_y1': bbox_y1,
        'bbox_y2': bbox_y2,
        'class': class_name,
        'fname': fname
    }
)

train_df.to_csv('dataset/car_train.csv', index=False)

test_bbox_x1 = []
test_bbox_x2 = []
test_bbox_y1 = []
test_bbox_y2 = []
test_fname = []

for k, v in train_mat.items():
    for annot in np.transpose(v[0]):
        for i, a in enumerate(annot):
            if i == 0:
                test_bbox_x1.append(np.asscalar(a))
            elif i == 1:
                test_bbox_y1.append(np.asscalar(a))
            elif i == 2:
                test_bbox_x2.append(np.asscalar(a))
            elif i == 3:
                test_bbox_y2.append(np.asscalar(a))
            elif i == 4:
                test_fname.append(np.asscalar(a))
                
test_df = pd.DataFrame(
    {
        'bbox_x1': test_bbox_x1,
        'bbox_x2': test_bbox_x2,
        'bbox_y1': test_bbox_y1,
        'bbox_y2': test_bbox_y2,
        'fname': test_fname
    }
)

test_df.to_csv('dataset/car_test.csv', index=False)