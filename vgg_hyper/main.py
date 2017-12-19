from __future__ import print_function
import argparse

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from model import VGG_16
#from vgg16 import VGG16
import os

from hyperspace import hyperdrive


batch_size = 32
num_classes = 10
epochs = 5
data_augmentation = False
num_predictions = 20

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


    # Train the model
    model.fit(x_train, y_train,
              batch_size=128,
              shuffle=True,
              epochs=250,
              validation_data=(x_test, y_test))

    # Evaluate the model
    scores = model.evaluate(x_test, y_test)

    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])

def objective(params, final_iter=None):
    kernel1 = int(params[0])
    kernel2 = int(params[1])
    # kernel3 = params[2]
    # kernel4 = params[3]
    # kernel5 = params[4]
    # kernel6 = params[5]
    # kernel7 = params[6]
    # kernel8 = params[7]
    # kernel9 = params[8]
    # kernel10 = params[9]
    # dropout1 = params[10]
    # dropout2 = params[11]

    model = VGG_16(kernel1, kernel2)
    #model = VGG16()
    model.compile(optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_train, y_train),
                            workers=4)

    # Save model and weights
    if final_iter:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = model.evaluate(x_train, y_train, verbose=1)
    print('Train loss:', scores[0])
    print('Train accuracy:', scores[1])

    return scores[0]


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    # hparams = [(2, 5),     # kernel1
    #            (2, 5),     # kernel2
    #            (2, 5),     # kernel3
    #            (2, 5),     # kernel4
    #            (2, 5),     # kernel5
    #            (2, 5),     # kernel6
    #            (2, 5),     # kernel7
    #            (2, 5),     # kernel8
    #            (2, 5),     # kernel9
    #            (2, 5),     # kernel10
    #            (0.5, 0.8), # dropout1
    #            (0.5, 0.8)] # dropout1

    hparams = [(2, 5), (2, 5)]


    hyperdrive(objective=objective,
               hyperparameters=hparams,
               results_path=args.results_dir,
               model="GP",
               n_iterations=11,
               verbose=True,
               random_state=0)


if __name__ == '__main__':
    main()
