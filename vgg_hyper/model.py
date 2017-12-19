import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import numpy as np

def VGG_16(kernel1=3, kernel2=3, kernel3=3, kernel4=3, kernel5=3, kernel6=3,
           kernel7=3, kernel8=3, kernel9=3, kernel10=3, dropout1=0.5, dropout2=0.5):
    """
    Defines the VGG16 model.
    """
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=(32, 32, 3)))
    model.add(Convolution2D(64, kernel_size=(kernel1, kernel1), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, kernel_size=(kernel2, kernel2), activation='relu'))
    model.add(MaxPooling2D((1,1), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(kernel3, kernel3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, kernel_size=(kernel4, kernel4), activation='relu'))
    model.add(MaxPooling2D((1,1), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kernel5, kernel5), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kernel6, kernel6), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, kernel_size=(kernel7, kernel7), activation='relu'))
    model.add(MaxPooling2D((1,1), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kernel8, kernel8), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kernel9, kernel9), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((1,1), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, kernel_size=(kernel10, kernel10), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((1,1), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout1))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout2))
    model.add(Dense(5000, activation='softmax'))

    return model
