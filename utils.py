from matplotlib import pyplot
from keras.datasets import mnist
from keras.utils import to_categorical
from numpy import mean
from numpy import std

class Utils():
    # load train and test dataset
    def load_handwriting_dataset():
        # load dataset
        (trainX, trainY), (testX, testY) = mnist.load_data()
        # reshape dataset to have a single channel
        trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
        testX = testX.reshape((testX.shape[0], 28, 28, 1))
        # one hot encode target values
        trainY = to_categorical(trainY).astype('float32')
        testY = to_categorical(testY).astype('float32')
        return trainX, trainY, testX, testY

    # scale pixels
    def prep_pixels(train, test):
        # convert from integers to floats
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')
        # normalize to range 0-1
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0
        # return normalized images
        return train_norm, test_norm