# import necessary packages
import matplotlib.pyplot as plt
import time

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os

from PIL import Image

import warnings
warnings.filterwarnings("ignore")

start = time.time()


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# load test data info
test_file = 'test_labels.csv'
test_label = pd.read_csv( test_file )
print('There are ', test_label.shape[0], ' testing images in ', len(test_label.label.unique()), ' classes.')
#test_label.head()

test_0 = test_label.loc[test_label['label'] == 0]
print(test_0.shape)
test_1 = test_label.loc[test_label['label'] == 1]
print(test_1.shape)

test = [test_0, test_1]
classes = len(test_label.label.unique())
# Specify the path to the directory containing the images
test_path = '/jet/home/thanhngp/HW2-CNN/test/test'

try:
    os.mkdir("/jet/home/thanhngp/HW2-CNN/test/test_resize")
except:
    print("Folder already found")

# create a folder to store resized image on drive
new_size = (128, 128)

for c in range(classes):
    try:
        os.mkdir("/jet/home/thanhngp/HW2-CNN/test/test_resize/" + str(c))
    except:
        print("Folder already found")

    test_dir = '/jet/home/thanhngp/HW2-CNN/test/test_resize/' + str(c)

    # print(train[c].img_name[0:10])
    # Loop over the images in the directory
    i = 0 
    for f in test[c].img_name: #(os.listdir(train_path)[0:100]):
        # Open the image using PIL
        path = os.path.join(test_path, f)
        # print(path)
        image = Image.open(path)

        # Output the size of the image
        #print("Image size: ", image.size)

        # Resize the image to a specific size
        resized_image = image.resize(new_size)
        #print("Image size: ", resized_image.size)

        # Save the resized image
        resized_image.save(os.path.join(test_dir, f))

        i += 1
        if i%100 == 0:
          print(i)
          
end1 = time.time()
print('time taken: ', end1 - start)


# load train data info
train_file = '/jet/home/thanhngp/HW2-CNN/train_labels.csv'
train_label = pd.read_csv( train_file )
print('There are ', train_label.shape[0], ' training images in ', len(train_label.label.unique()), ' classes.')
train_label.head()

train_0 = train_label.loc[train_label['label'] == 0]
print(train_0.shape)
train_1 = train_label.loc[train_label['label'] == 1]
print(train_1.shape)

train = [train_0, train_1]
# Specify the path to the directory containing the images
train_path = '/jet/home/thanhngp/HW2-CNN/train/train'

try:
    os.mkdir("/jet/home/thanhngp/HW2-CNN/train/train_resize")
except:
    print("Folder already found")

# create a folder to store resized image on drive
new_size = (128, 128)

for c in range(classes):
    try:
        os.mkdir("/jet/home/thanhngp/HW2-CNN/train/train_resize/" + str(c))
    except:
        print("Folder already found")

    train_dir = '/jet/home/thanhngp/HW2-CNN/train/train_resize/' + str(c)

    # print(train[c].img_name[0:10])
    # Loop over the images in the directory
    i = 0
    for f in train[c].img_name: #(os.listdir(train_path)[0:100]):
        # Open the image using PIL
        path = os.path.join(train_path, f)
        #print(path)
        image = Image.open(path)

        # Output the size of the image
        #print("Image size: ", image.size)

        # Resize the image to a specific size
        resized_image = image.resize(new_size)
        #print("Image size: ", resized_image.size)

        # Save the resized image
        resized_image.save(os.path.join(train_dir, f))

        i += 1
        if i%1000 == 0:
          print(i)

end2 = time.time()
print('time taken: ', end2 - end1)