from __future__ import print_function
import argparse

import keras
from data_utils import load_data
from sklearn.model_selection import train_test_split

from model import vgg16
from hyperspace import hyperdrive


num_classes = 10
batch_size = 32
epochs = 5

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = load_data()
# Further split to create validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=x_test.shape[0],
                                                  shuffle=True, random_state=0)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_val /= 255
x_test /= 255


def objective(params):
    """
    Objective function to be minimized.

    Parameters
    ----------
    `params` [list]
        Hyperparameters to be set in optimization iteration.
        - Managed by hyperdrive.
    """
    kernel1 = int(params[0])
    kernel2 = int(params[1])
    # kernel3 = int(params[2])
    # kernel4 = int(params[3])
    # kernel5 = int(params[4])
    # kernel6 = int(params[5])
    # batch_size = int(params[6])

    # model = vgg16(kernel1=kernel1, kernel2=kernel2, kernel3=kernel3,
    #               kernel4=kernel4, kernel5=kernel5, kernel6=kernel6)
    model = vgg16(kernel1=kernel1, kernel2=kernel2)

    model.compile(optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_val, y_val),
              shuffle=True)

    # Score trained model.
    scores = model.evaluate(x_val, y_val, verbose=1)
    print('Validation loss:', scores[0])
    print('Validation accuracy:', scores[1])

    return scores[0]


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    hparams = [(2, 8),     # kernel1
               (2, 8)]     # kernel2
               # (2, 8),     # kernel3
               # (2, 8),     # kernel4
               # (2, 8),     # kernel5
               # (2, 8),     # kernel6
               # (32, 64)]   # batch_size

    hyperdrive(objective=objective,
               hyperparameters=hparams,
               results_path=args.results_dir,
               model="GP",
               n_iterations=11,
               verbose=True,
               random_state=0)


if __name__ == '__main__':
    main()
