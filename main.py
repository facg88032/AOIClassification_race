from CNN import cnn
from keras.optimizers import  Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
def load_data(path):
    data=np.load(path)
    return data
def read_csv(csv_path):
    df = pd.read_csv(csv_path, skiprows=1, header=None )
    Label=df[1]
    return Label


def get_dataset(data_path,csv_path):
    X=load_data(data_path)
    Y=read_csv(csv_path)
    return X,Y
def get_preprocessed_dataset(X,Y):
    Y=pd.get_dummies(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,shuffle = True)
    return x_train , y_train , x_test , y_test


def ImageGenerator(path):
    train_gen = ImageDataGenerator(
        rotation_range=40,
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        #vertical_flip=True,
        horizontal_flip=True,
        fill_mode='nearest',
        # featurewise_center=True,
        # featurewise_std_normalization=True
        validation_split = 0.2
        )
    train_generator = train_gen.flow_from_directory(
            path,  # this is the target directory
            target_size=(256, 256),  # all images will be resized to 256*256
            batch_size=16,
            class_mode='categorical',   # since we use binary_crossentropy loss, we need binary labels
            shuffle=True,
            subset='training')
    validation_generator = train_gen.flow_from_directory(
        path,
        target_size=(256, 256),
        batch_size=16,
        class_mode='categorical',
        shuffle=True,
        subset = 'validation')
    return train_generator,validation_generator


def generate_optimizer():
    lr = 0.0001
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-08

    return Adam( lr=lr ,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon)

def train(model,train_gen,vaild_gen):
    model.compile(loss='categorical_crossentropy', optimizer= generate_optimizer(), metrics=['accuracy'])
    filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]
    model.fit_generator(
        train_gen,
        steps_per_epoch = train_gen.samples//16,
        validation_data = vaild_gen,
        validation_steps=vaild_gen.samples // 16,
        epochs=1000,
        verbose=1,
        callbacks=callbacks_list)
    return model

def test(model,x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    print('Test:')
    print('Loss: %s\nAccuracy: %s' % (loss, accuracy))



def main():
    # data_path='x_train.npy'
    # csv_path= '../Aidata/aoi_race/train.csv'
    #Images , Labels= get_dataset(data_path,csv_path)
    #x_train, y_train, x_test, y_test=get_preprocessed_dataset(Images,Labels)
    train_data_dir ='../Aidata/aoi_race/train_images'
    train_gen,vaild_gen=ImageGenerator(train_data_dir)
    model=cnn()
    model=model.InceptionV1(256,256,3,6)
    train(model,train_gen, vaild_gen)
    #test(model,x_test,y_test)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
