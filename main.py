import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.utils import np_utils, plot_model
from keras.datasets import mnist
from keras.optimizers import  Adam
import matplotlib.pyplot as plt
import pandas as pd


def get_dataset():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    return X_train, Y_train, X_test, Y_test

def get_preprocessed_dataset():
    X_train, Y_train, X_test, Y_test = get_dataset()
    x_train = X_train.reshape(60000, 1, 28, 28) / 255
    x_test = X_test.reshape(10000, 1, 28, 28) / 255
    y_train = np_utils.to_categorical(Y_train)
    y_test = np_utils.to_categorical(Y_test)
    return x_train , y_train , x_test , y_test

def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, input_shape=(1, 28, 28), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=2, data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    print(model.summary())
    return  model

def generate_optimizer():
    return Adam()

def train(model,x_train, y_train):
    model.compile(loss='categorical_crossentropy', optimizer= generate_optimizer(), metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1)
    return model

def test(model,x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    print('Test:')
    print('Loss: %s\nAccuracy: %s' % (loss, accuracy))



def main():
    x_train, y_train, x_test, y_test=get_preprocessed_dataset()
    model=build_model()
    model=train(model,x_train, y_train)
    test(model,x_test,y_test)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
