from __future__ import absolute_import
import sys
from six.moves import cPickle

from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
import os

from sklearn.model_selection import train_test_split


def load_data(path='../data/cifar-10-batches-py/'):
    """
    Loads CIFAR10 dataset.

    Parameters
    ----------
    * `path`: [str, default='../data/cifar-10-batches-py/']
        Path to data batches
    Returns
    -------
    * Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def load_batch(fpath, label_key='labels'):
    """
    Internal utility for parsing CIFAR data.

    Parameters
    ----------
    * `fpath`: [str]
        Path the file to parse.
    * `label_key`:
        key for label data in the retrieve dictionary.

    Returns
    -------
    * A tuple `(data, labels)`.
    """
    print("loading data from {}\n".format(fpath))

    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)

    return data, labels


# (x_train, y_train), (x_test, y_test) = load_data()
# print("Data loaded.")
# print("x_train shape before splitting: {}\n".format(x_train.shape))
#
# x_train, y_train, x_val, y_val = train_test_split(x_train, y_train, test_size=x_test.shape[0],
#                                                   shuffle=True, random_state=0)
#
# print("x_train shape after splitting: {}\n".format(x_train.shape))
# print("y_train shape after splitting: {}\n".format(y_train.shape))
# print("x_val shape after splitting: {}\n".format(x_val.shape))
# print("y_val shape after splitting: {}\n".format(y_val.shape))
#
# print("x_test shape: {}\n".format(x_test.shape))
# print("y_test shape: {}\n".format(y_test.shape))
