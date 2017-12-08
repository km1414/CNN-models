"""
Helper functions for CIFAR-10 data preparation
Adapted from: https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/cifar10.py
"""

import numpy as np
import pylab as pl
import pickle
import os
import download
from keras.utils.np_utils import to_categorical


data_path = "data/CIFAR-10/"
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 10
_num_files_train = 5
_images_per_file = 10000
_num_images_train = _num_files_train * _images_per_file


def _get_file_path(filename=""):

    return os.path.join(data_path, "cifar-10-batches-py/", filename)


def _unpickle(filename):

    # Create full path for the file.
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:

        data = pickle.load(file, encoding='bytes')

    return data


def _convert_images(raw):

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


def _load_data(filename):

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])

    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls


def download_and_extract_data():

    download.download(url=data_url, path=data_path, kind='tar.gz', progressbar=True, replace=False, verbose=True)


def load_class_names():

    # Load the class-names from the pickled file.
    raw = _unpickle(filename="batches.meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names


def load_training_data():

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls, to_categorical(cls, num_classes=num_classes)


def load_test_data():

    images, cls = _load_data(filename="test_batch")

    return images, cls, to_categorical(cls, num_classes=num_classes)


# Plot results
def plot_results(results):
    pl.figure()

    pl.subplot(121)
    pl.plot(results.history['dense_5_acc'])
    pl.title('Accuracy:')
    pl.plot(results.history['val_dense_5_acc'])
    pl.legend(('Train', 'Validation'))

    pl.subplot(122)
    pl.plot(results.history['loss'])
    pl.title('Cost:')
    pl.plot(results.history['val_loss'])
    pl.legend(('Train', 'Validation'))
