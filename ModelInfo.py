# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 22:14:24 2022

@author: Santosh
"""
import  os
import glob

import cv2 as cv

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def read_image_resize(dataset_folder):
    
    images =[]
    labels =[]
    for directory in os.listdir(dataset_folder):
        print(directory)
        
        path_dir = os.path.join(dataset_folder, directory)
        
        print(path_dir)
        
        all_bmp_images = glob.glob(path_dir + "**/*.jpg")
        
        print(all_bmp_images)
        for bmp_image in all_bmp_images:
            #print(bmp_image)
            image = cv.imread(bmp_image, cv.IMREAD_COLOR)

            images.append(image)
            
            labels.append(directory)
            
    return images, labels


def create_sequential(no_of_classes, x_train):

    model = Sequential()
    
    # FIRST LAYER
    model.add(Dense(units=20, activation='relu', input_shape=x_train[0].shape))
   
    #model.add(Dense(units=20, activation='relu'))

    model.add(Dense(units=10, activation='relu'))

    model.add(Dense(no_of_classes, activation='softmax'))
    
    model.compile( optimizer='adam',
                   loss = 'sparse_categorical_crossentropy',
                   metrics = 'accuracy')
    
    return model

def create_sequentia2(no_of_classes, x_train):

    model = Sequential()
    
    # FIRST LAYER
    model.add(Dense(units=18, activation='relu', input_shape=x_train[0].shape))
   
    #model.add(Dense(units=20, activation='relu'))

    model.add(Dense(units=9, activation='relu'))

    model.add(Dense(no_of_classes, activation='softmax'))
    
    model.compile( optimizer='adam',
                   loss = 'sparse_categorical_crossentropy',
                   metrics = 'accuracy')
    
    return model

def encode_labels(y_train):
    y_encode_train = []
    for x_t in y_train:
        print(x_t)
        if(x_t=='ANGRY'):
           y_encode_train.append(0)
        if(x_t=='HAPPY'):
           y_encode_train.append(1)  
        if(x_t=='SAD'):
           y_encode_train.append(2) 
           
    return y_encode_train


def decode_labels(output):
    index = np.argmax(output)
    class_name = 'NORMAL'
    if index==0:
       class_name='ANGRY'
    if index==1:
       class_name='HAPPY'  
    if index==2:
       class_name='SAD'  
           
    return class_name, index


def convert_to_tensor(x_train, y_train):
    x_train = np.array(x_train, dtype=object) 
    x_train = np.asarray(x_train).astype(np.float32)
    x_train = tf.convert_to_tensor(x_train)


    y_encode_train = encode_labels(y_train)
    y_train = np.array(y_encode_train, dtype=np.int32) 
    y_train = tf.convert_to_tensor(y_train)
    
    return x_train, y_train

def convert_to_tensor_(x_train):
    x_train = np.array(x_train, dtype=object) 
    x_train = np.asarray(x_train).astype(np.float32)
    x_train = tf.convert_to_tensor(x_train)
    
    return x_train

def reshape_array(angle_features):
    angle_features = np.array(angle_features, dtype=object) 
    angle_features = np.asarray(angle_features).astype(np.float32)
    angle_features_r = angle_features.reshape(1, angle_features.shape[0])

    feature_tensor_out= convert_to_tensor_(angle_features_r)
    
    return feature_tensor_out
    
def plot_model_summary(model):

    history = model.history.history

    import matplotlib.pyplot as plt
    
    plt.subplot(2, 1, 1)
    plt.title("MODEL LOSS")
    plt.ylabel("Loss")
    plt.xlabel("No of Epochs")

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])

    plt.legend(['train', 'test'], loc='lower right')


    plt.subplot(2, 1, 2)
    plt.title("MODEL Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("No of Epochs")

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])

    plt.legend(['train', 'test'], loc='lower right')

    plt.tight_layout()
                       