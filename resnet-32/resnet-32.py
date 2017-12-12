"""
ResNet-32 model for CIFAR-10 image recognition.

Inspired by:
    Deep Learning Specialization at Coursera
    https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

Model architecture for CIFAR-10 dataset classification reproduced from original paper:
    https://arxiv.org/pdf/1512.03385.pdf

Karolis M. December 2017.
"""

from helpers import *
from keras.layers import *
from keras.models import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


# Data preparation
download_and_extract_data()
x_train, _, y_train= load_training_data()
x_test, _, y_test = load_test_data()
input_shape = list(x_train.shape[1:])
classes = y_train.shape[1]


# ResNet building block of two layers
def building_block(X, filter_size, filters, stride=1):

    # Save the input value for shortcut
    X_shortcut = X

    # Reshape shortcut for later adding if dimensions change
    if stride > 1:

        X_shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    # First layer of the block
    X = Conv2D(filters, kernel_size = filter_size, strides=stride, padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Second layer of the block
    X = Conv2D(filters, kernel_size = filter_size, strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = add([X, X_shortcut])  # Add shortcut value to main path
    X = Activation('relu')(X)

    return X


# Full model
def create_model(input_shape, classes, name):

    # Define the input
    X_input = Input(input_shape)

    # Stage 1
    X = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same')(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Stage 2
    X = building_block(X, filter_size=3, filters=16, stride=1)
    X = building_block(X, filter_size=3, filters=16, stride=1)
    X = building_block(X, filter_size=3, filters=16, stride=1)
    X = building_block(X, filter_size=3, filters=16, stride=1)
    X = building_block(X, filter_size=3, filters=16, stride=1)

    # Stage 3
    X = building_block(X, filter_size=3, filters=32, stride=2)  # dimensions change (stride=2)
    X = building_block(X, filter_size=3, filters=32, stride=1)
    X = building_block(X, filter_size=3, filters=32, stride=1)
    X = building_block(X, filter_size=3, filters=32, stride=1)
    X = building_block(X, filter_size=3, filters=32, stride=1)

    # Stage 4
    X = building_block(X, filter_size=3, filters=64, stride=2)  # dimensions change (stride=2)
    X = building_block(X, filter_size=3, filters=64, stride=1)
    X = building_block(X, filter_size=3, filters=64, stride=1)
    X = building_block(X, filter_size=3, filters=64, stride=1)
    X = building_block(X, filter_size=3, filters=64, stride=1)

    # Average pooling and output layer
    X = GlobalAveragePooling2D()(X)
    X = Dense(classes, activation='softmax')(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name=name)

    return model


# Define optimizer and compile model
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ResNet32 = create_model(input_shape=input_shape, classes=classes, name='ResNet32')
ResNet32.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])


# Generator for data augmantation
datagen = ImageDataGenerator(
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True)  # randomly flip images


# Train model
results = ResNet32.fit_generator(datagen.flow(x_train, y_train,
                                 batch_size = 250),
                                 epochs = 100,
                                 steps_per_epoch=200,  # data_size/batch size
                                 validation_data=(x_test, y_test))

# Plot train / validation results
plot_results(results)

# Print model architecture
ResNet32.summary()


