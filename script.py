import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob as gb
import cv2
import tensorflow as tf
import keras
import os
from sklearn.metrics import confusion_matrix
from contextlib import redirect_stdout

# Dataset Summary
dataset_summary = """
## Dataset Summary

### This dataset contains over 14,000 images that need to be classified into 6 distinct categories. Here’s a quick breakdown:
- Total Images: 14,000+
- Classes: 6 categories, each representing a unique label for classification.
- Image Format: Likely standardized (e.g., JPEG, PNG).
- Size and Quality: Varies, but typically consistent for training uniformity.
- Typical Categories in Image Classification

### While not specified, common image classification categories could include:
- Natural Scenes: Different environments (e.g., mountains, forests).
- Urban and Rural Scenes: Different landscapes (e.g., streets, buildings).
- Objects: Specific items within scenes.

#### Use Case:

The model developed from this dataset would likely employ CNNs (Convolutional Neural Networks) due to their effectiveness in image feature extraction and spatial hierarchy, aiming for a high accuracy similar to 98% as mentioned in your Intel classification example.
"""

# Import Data And Preprocessing
train_path = 'archive/seg_train/'
test_path = 'archive/seg_test/'
pred_path = 'archive/seg_pred/'

def open_folders(path, file, name='Training Data'):
    for folder in os.listdir(path + file):
        files = gb.glob(pathname=path + file + '/' + folder + '/*.jpg')
        print(f'For {name} : Found {len(files)} images in folder {folder}')

def get_image_size(path, folder_name):
    size = []
    if folder_name != 'seg_pred':
        for folder in os.listdir(path + folder_name):
            files = gb.glob(pathname=path + folder_name + '/' + folder + '/*.jpg')
            for file in files:
                image = plt.imread(file)
                size.append(image.shape)
    else:
        files = gb.glob(pathname=path + folder_name + '/*.jpg')
        for file in files:
            image = plt.imread(file)
            size.append(image.shape)
    
    print(pd.Series(size).value_counts())

def resize_images(images, size=(100, 100)):
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, size)
        resized_images.append(resized_img)
    return resized_images

def plot_confusion_matrix(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.tight_layout()
    plt.show()

def main():
    with open('output.txt', 'w') as f:
        with redirect_stdout(f):
            print(dataset_summary)
            
            print('-' * 40 + ' Training Data ' + '-' * 46)
            open_folders(train_path, 'seg_train')
            print('\n' + '-' * 40 + ' Test Data ' + '-' * 50)
            open_folders(test_path, 'seg_test', name='Test Data')
            print('\n' + '-' * 40 + ' Prediction Data ' + '-' * 44)
            files = gb.glob(pathname=pred_path + 'seg_pred' + '/*.jpg')
            print(f'For Prediction Data : Found {len(files)} images in folder Prediction')
            
            print('-' * 40 + ' Training Data ' + '-' * 46)
            get_image_size(train_path, 'seg_train')
            print('\n' + '-' * 40 + ' Test Data ' + '-' * 50)
            get_image_size(test_path, 'seg_test')
            print('\n' + '-' * 40 + ' Prediction Data ' + '-' * 44)
            get_image_size(pred_path, 'seg_pred')
            
            print('-' * 40 + ' Training Data ' + '-' * 46)
            open_folders(train_path, 'seg_train')
            print('\n' + '-' * 40 + ' Test Data ' + '-' * 50)
            open_folders(test_path, 'seg_test', name='Test Data')
            print('\n' + '-' * 40 + ' Prediction Data ' + '-' * 44)
            files = gb.glob(pathname=pred_path + 'seg_pred' + '/*.jpg')
            print(f'For Prediction Data : Found {len(files)} images in folder Prediction')
            
            code = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}
            
            def getcode(n):
                for x, y in code.items():
                    if n == y:
                        return x
            
            x_train = []
            y_train = []
            
            for folder in os.listdir(train_path + 'seg_train'):
                files = gb.glob(pathname=train_path + 'seg_train/' + folder + '/*.jpg')
                for file in files:
                    image = cv2.imread(file)
                    image_array = cv2.resize(image, (150, 150))
                    x_train.append(list(image_array))
                    y_train.append(code[folder])
            
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            
            x_test = []
            y_test = []
            
            for folder in os.listdir(test_path + 'seg_test'):
                files = gb.glob(pathname=test_path + 'seg_test/' + folder + '/*.jpg')
                for file in files:
                    image = cv2.imread(file)
                    image_array = cv2.resize(image, (150, 150))
                    x_test.append(list(image_array))
                    y_test.append(code[folder])
            
            x_test = np.array(x_test)
            y_test = np.array(y_test)
            
            x_pred = []
            y_pred = []
            
            files = gb.glob(pathname=pred_path + 'seg_pred' + '/*.jpg')
            for file in files:
                image = cv2.imread(file)
                image_array = cv2.resize(image, (150, 150))
                x_pred.append(list(image_array))
                y_pred.append(file.split('/')[-1])
            
            x_pred = np.array(x_pred)
            y_pred = np.array(y_pred)
            
            # Normalize the data
            x_train = x_train / 255
            x_test = x_test / 255
            x_pred = x_pred / 255
            
            print('-' * 40 + ' Training Data ' + '-' * 46)
            print(f'We Have {len(x_train)} Image In X_train')
            print(f'We Have {len(y_train)} items In y_train ')
            
            print('\n' + '-' * 40 + ' Test Data ' + '-' * 50)
            print(f'We Have {len(x_test)} Image In X_test')
            print(f'We Have {len(y_test)} items In y_test')
            
            print('\n' + '-' * 40 + ' Prediction Data ' + '-' * 44)
            print(f'We Have {len(x_pred)} Image In X_pred')
            
            X_train, y_train = np.array(x_train), np.array(y_train)
            X_test, y_test = np.array(x_test), np.array(y_test)
            X_pred = np.array(x_pred)
            
            print(f'X_train shape is {X_train.shape}')
            print(f'X_test shape is {X_test.shape}')
            print(f'y_train shape is {y_train.shape}')
            print(f'y_test shape is {y_test.shape}')
            print(f'X_pred shape is {X_pred.shape}')

            # build CNN model
            cnnModel = keras.models.Sequential([
                keras.layers.Conv2D(256, kernel_size = (3, 3), activation='relu', input_shape = (100, 100, 3)), # feature map -> (98, 98, 256)
                keras.layers.Conv2D(128, kernel_size = (3, 3), activation = 'relu'), # feature map -> (96, 96, 128)
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(3, 3), # feature map -> (32, 32, 128)
                keras.layers.Conv2D(64, kernel_size = (3, 3), activation='relu'), # feature map -> (30, 30, 64)
                keras.layers.Conv2D(32, kernel_size=(3, 3), activation= 'relu'),  # feature map -> (28, 28, 32)
                keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu'), # feature map -> (26, 26, 16)
                keras.layers.BatchNormalization(),
                keras.layers.MaxPool2D(3, 3),  # feature map -> (8, 8, 16) 
                
                keras.layers.Flatten(),  # 1024
                keras.layers.Dense(128, activation = 'relu'), 
                keras.layers.Dense(64, activation = 'relu'),
                keras.layers.Dense(32, activation = 'relu'),
                keras.layers.Dense(6, activation='softmax')
            ])

            print(cnnModel.summary())

            cnnModel.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
            cnnModel.fit(X_train, y_train, epochs = 25, batch_size=64, verbose=1)

            loss, accuracy = cnnModel.evaluate(X_test, y_test)
            print('Test Loss is {}'.format(loss))
            print('Test Accuracy is {}'.format(accuracy ))

            y_test_pred = cnnModel.predict(X_test)
            print(f'Prediction Shape is {y_test_pred.shape}')

            class_names = ['buildings','forest','glacier','mountain','sea','street' ]   
            pred_labels = np.argmax(y_test_pred, axis=1)  
            plot_confusion_matrix(y_test, pred_labels, class_names)

if __name__ == "__main__":
    main()