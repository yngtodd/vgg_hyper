from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


def vgg16(kernel1=3, kernel2=3, kernel3=3, kernel4=3, kernel5=3, kernel6=3,
          kernel7=3, kernel8=3, kernel9=3, kernel10=3, dropout1=0.25,
          dropout2=0.25, dropout3=0.25, dropout4=0.25, dropout5=0.25, dropout6=0.25):
    """
    VGG16.

    Parameters
    ----------
    `kernel*` [int, default=3]
        Convolution kernel size.

    `dropout*` [float, default=0.25]
        Dropout proportion at the end of each block.

    Returns
    -------
    `model` [keras.models.Sequential()]
    """
    model = Sequential()

    # Block 1
    model.add(Conv2D(32, kernel_size=(kernel1, kernel1), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, kernel_size=(kernel2, kernel2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(dropout1))

    # Block 2
    model.add(Conv2D(128, kernel_size=(kernel3, kernel3), padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=(kernel4, kernel4), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(dropout2))

    # Block 3
    model.add(Conv2D(256, kernel_size=(kernel5, kernel5), padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=(kernel6, kernel6), padding='same', activation='relu'))
    model.add(Conv2D(256, kernel_size=(kernel7, kernel7), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(dropout3))

    # Block 4
    model.add(Conv2D(512, kernel_size=(kernel8, kernel8), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(kernel9, kernel9), padding='same', activation='relu'))
    model.add(Conv2D(512, kernel_size=(kernel10, kernel10), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(dropout4))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout6))
    model.add(Dense(10, activation='softmax'))

    return model
