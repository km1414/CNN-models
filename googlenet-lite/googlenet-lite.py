"""
GoogLeNet-lite (Inception) model for CIFAR-10 image recognition.

Inspired by:
    Deep Learning Specialization at Coursera
    http://dandxy89.github.io/ImageModels/googlenet/#info

GoogLeNet architecture adapted for CIFAR-10 dataset classification. Original paper:
    https://arxiv.org/pdf/1409.4842.pdf

Karolis M. December 2017.
"""


from helpers import *
from keras.layers import *
from keras.models import Model
from keras import optimizers
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import gc


# Data preparation
download_and_extract_data()
x_train, _, y_train= load_training_data()
x_test, _, y_test = load_test_data()
input_shape = list(x_train.shape[1:])
classes = y_train.shape[1]



# Inception module - main building block
def inception_module(X, filter_sizes):

    # 1x1 covolution
    conv_1x1 = Conv2D(filter_sizes[0], kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(X)

    # Bottleneck layer and 3x3 convolution
    conv_3x3 = Conv2D(filter_sizes[1], kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(X)
    conv_3x3 = Conv2D(filter_sizes[2], kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(conv_3x3)

    # Bottleneck layer and 5x5 convolution
    conv_5x5 = Conv2D(filter_sizes[3], kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(X)
    conv_5x5 = Conv2D(filter_sizes[4], kernel_size=5, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(conv_5x5)

    # Max pooling and bottleneck layer
    max_pool = MaxPooling2D(pool_size=3, strides=1, padding='same')(X)
    max_pool = Conv2D(filter_sizes[5], kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(max_pool)

    # Concatenate all tensors to 1 tensor
    X = concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool], axis=3)

    return X



# Auxiliary classifier - for predictions in a middle stage
def aux_classifier(X):

    # Average pooling, fc, dropout, fc
    X = AveragePooling2D(pool_size=3, strides=2, padding='same')(X)
    X = Conv2D(filters=128, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(0.0002))(X)
    X = Flatten()(X)
    X = Dense(1024, activation='relu', kernel_regularizer=l2(0.0002))(X)
    X = Dropout(0.7)(X)
    X = Dense(classes, activation='softmax', kernel_regularizer=l2(0.0002))(X)

    return X



# Full model
def create_model(input_shape, classes):

    # Define the input
    X_input = Input(input_shape)

    # Stage 1 - layers before inception modules
    X = Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(X_input)
    X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Conv2D(filters=64, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(0.0002))(X)
    X = Conv2D(filters=192, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(X)
    X = BatchNormalization(axis=3)(X)
    X = MaxPooling2D(pool_size=3, strides=1, padding='same')(X)

    # Stage 2 - 2 inception modules and max pooling
    X = inception_module(X, filter_sizes=[64, 96, 128, 16, 32, 32])
    X = inception_module(X, filter_sizes=[128, 128, 192, 32, 96, 64])
    X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)

    # Stage 3 - 5 inception modules and max pooling
    X = inception_module(X, filter_sizes=[192, 96, 208, 16, 48, 64])
    aux_output_1 = aux_classifier(X)   # Auxiliary classifier
    X = inception_module(X, filter_sizes=[160, 112, 225, 24, 64, 64])
    X = inception_module(X, filter_sizes=[128, 128, 256, 24, 64, 64])
    X = inception_module(X, filter_sizes=[112, 144, 288, 32, 64, 64])
    aux_output_2 = aux_classifier(X) # Auxiliary classifier
    X = inception_module(X, filter_sizes=[256, 160, 320, 32, 128, 128])
    X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)

    # Stage 4 - 2 inception modules and average pooling
    X = inception_module(X, filter_sizes=[256, 160, 320, 32, 128, 128])
    X = inception_module(X, filter_sizes=[384, 192, 384, 48, 128, 128])
    X = AveragePooling2D(pool_size=4, strides=1, padding='valid')(X)

    # Stage 5 - dropout, linear fc, softmax fc
    X = Flatten()(X)
    X = Dropout(0.4)(X)
    X_output = Dense(classes, activation='softmax', kernel_regularizer=l2(0.0002))(X)

    # Create model - combine main classifier with auxiliary classifiers
    model = Model(inputs=X_input, outputs=[X_output, aux_output_1, aux_output_2])

    return model


# Define optimizer and compile model
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
GoogLeNet = create_model(input_shape=input_shape, classes=classes)
# loss_weights - weighting losses of main classifier and auxiliary classifiers
GoogLeNet.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'], loss_weights=[1., 0.3, 0.3])


# Generator for data augmantation
datagen = ImageDataGenerator(
            rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True,
            zoom_range=0.1)


history = {'train_acc':[], 'train_loss':[], 'test_acc':[], 'test_loss':[]}
epochs = 100


# Training loop
for epoch in range(epochs):

    print('Epoch:', epoch)
    # Processing data augmentation on full data set
    x_train_aug, y_train_aug = datagen.flow(x_train, y_train, batch_size=len(x_train), shuffle=True).next()
    # Free unused memory
    _ = gc.collect()
    # Train 1 epoch
    results = GoogLeNet.fit(x=x_train_aug, y=[y_train_aug, y_train_aug, y_train_aug],
                            epochs=1,
                            batch_size=250,
                            validation_data=(x_test, [y_test, y_test, y_test]))

    # Save epoch results
    history['train_acc'].append(results.history['dense_5_acc'][0])
    history['train_loss'].append(results.history['dense_5_loss'][0])
    history['test_acc'].append(results.history['val_dense_5_acc'][0])
    history['test_loss'].append(results.history['val_dense_5_acc'][0])

    # Schedule learning rate changes
    if (epoch % 200 == 1) or (epoch % 250 == 1):
        lr = K.eval(GoogLeNet.optimizer.lr)
        lr = K.eval(GoogLeNet.optimizer.lr.assign(lr / 4))


# Plot train / validation results
plot_results(history)

# Print model architecture
GoogLeNet.summary()


