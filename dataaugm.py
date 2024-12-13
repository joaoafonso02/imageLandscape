## Import Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import glob as gb
import cv2
import tensorflow as tf
import keras
import os
from sklearn.metrics import confusion_matrix, classification_report

## Import Data And Preprocessing
train_path = 'archive/seg_train/'
test_path = 'archive/seg_test/'
pred_path = 'archive/seg_pred/'

def open_folders(path, file, name = 'Traning Data'):
    for folder in os.listdir(path + file):
        files = gb.glob(pathname = path + file + '/' + folder + '/*.jpg')
        print(f'For {name} : Found {len(files)} images in folder {folder}')

print('-' * 40 + ' Traning Data ' + '-' * 46)
open_folders(train_path, 'seg_train')
print('\n' + '-' * 40 + ' Test Data ' + '-' * 50)
open_folders(test_path, 'seg_test', name = 'Test Data')
print('\n' +'-' * 40 + ' Prediction Data ' + '-' * 44)
files = gb.glob(pathname = pred_path + 'seg_pred' + '/*.jpg')
print(f'For Prediction Data : Found {len(files)} images in folder Prediction')

code = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}

# Get the labels for the images
def getcode(n):
    for x, y in code.items():
        if n == y:
            return x

new_size = 100

def get_image_array(path, folder_name, new_size = new_size):
    X = []
    y = []
    if folder_name != 'seg_pred':
        for folder in os.listdir(path + folder_name):
            files = gb.glob(pathname= path + folder_name + '/' + folder + '/*.jpg')
            for file in files:
                image = cv2.imread(file)
                image_array = cv2.resize(image, (new_size, new_size))
                X.append(image_array)
                y.append(code[folder])
    else:
        files = gb.glob(pathname= path + folder_name + '/*.jpg')
        for file in files:
            image = cv2.imread(file)
            image_array = cv2.resize(image, (new_size, new_size))
            X.append(image_array)
    return X, y

X_train, y_train = get_image_array(train_path, 'seg_train')
X_test, y_test = get_image_array(test_path, 'seg_test')
X_pred, _ = get_image_array(pred_path, 'seg_pred')

print('-' * 40 + ' Traning Data ' + '-' * 46)
print(f'We Have {len(X_train)} Image In X_train')
print(f'We Have {len(y_train)} items In y_train ')

print('\n' +'-' * 40 + ' Test Data ' + '-' * 50)
print(f'We Have {len(X_test)} Image In X_test')
print(f'We Have {len(y_test)} items In y_test')

print('\n' +'-' * 40 + ' Prediction Data ' + '-' * 44)
print(f'We Have {len(X_pred)} Image In X_pred')

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)
X_pred = np.array(X_pred)

print(f'X_train shape is {X_train.shape}') 
print(f'X_test shape is {X_test.shape}')
print(f'y_train shape is {y_train.shape}')
print(f'y_test shape is {y_test.shape}')
print(f'X_pred shape is {X_pred.shape}')

# Data Augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Build CNN Model
cnnModel = keras.models.Sequential([
    keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 3)), # feature map -> (98, 98, 256)
    keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'), # feature map -> (96, 96, 128)
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(3, 3), # feature map -> (32, 32, 128)
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'), # feature map -> (30, 30, 64)
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),  # feature map -> (28, 28, 32)
    keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu'), # feature map -> (26, 26, 16)
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(3, 3),  # feature map -> (8, 8, 16) 
    
    keras.layers.Flatten(),  # 1024
    keras.layers.Dense(128, activation='relu'), 
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(6, activation='softmax')
])

print(cnnModel.summary())
cnnModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model without data augmentation
history = cnnModel.fit(X_train, y_train, batch_size=64, epochs=25, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
loss, accuracy = cnnModel.evaluate(X_test, y_test)
print('Test Loss is {}'.format(loss))
print('Test Accuracy is {}'.format(accuracy))

# Generate classification report
y_test_pred = cnnModel.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
report = classification_report(y_test, y_test_pred_classes, target_names=code.keys())
print(report)

# Plot and save the loss function and accuracy plots
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plotClass.png')

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plotClass.png')

def plot_confusion_matrixPercentage(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Normalize by the number of true labels in each row to get percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Convert to percentage
    cm_percentage = np.nan_to_num(cm_percentage)  # Replace NaN with 0 if division by zero occurs

    # Create custom annotations with percentage symbol
    annotations = np.array([[f'{int(value)}%' for value in row] for row in cm_percentage])

    plt.figure(figsize=(8, 6))  
    sns.heatmap(cm_percentage, annot=annotations, fmt="", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names, cbar=False)
    
    plt.title("Confusion Matrix (Percentage)", fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.tight_layout()
    plt.savefig('matrixNo.png')
    plt.show()

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']   
plot_confusion_matrixPercentage(y_test, y_test_pred_classes, class_names)

# Plot and save sample predictions
plt.figure(figsize=(30, 40))
for n, i in enumerate(list(np.random.randint(0, len(X_test), 36))):
    plt.subplot(6, 6, n+1)
    plt.imshow(X_test[i])
    plt.axis('off')
    plt.title(f'Actual: {getcode(y_test[i])}\n Predict: {getcode(y_test_pred_classes[i])}', fontdict={'fontsize': 14, 'color': 'blue'})
plt.savefig('imagePredictionClass.png')