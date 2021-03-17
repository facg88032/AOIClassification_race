import keras
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def load_model(path):
    model=keras.models.load_model(path)
    return model
def read_csv(csv_path):
    df = pd.read_csv(csv_path, skiprows=1, header=None )
    return df
def write_csv(data):
    data.to_csv('test.csv',header=['ID','Label'] , index=0)

def ImageGenerator(path):
    test_gen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_gen.flow_from_directory(
        directory=path,  # this is the target directory
        target_size=(256, 256),  # all images will be resized to 256*256
        batch_size=1,
        class_mode=None,  # since we use binary_crossentropy loss, we need binary labels
        shuffle=False,

        )
    return  test_generator
def main():
    test_data_dir='../Aidata/aoi_race/test_images/'

    model_path='weights-improvement-336-0.99.hdf5'
    test_genX=ImageGenerator(test_data_dir)


    model=load_model(model_path)
    output=model.predict_generator(test_genX,
                                   steps=test_genX.samples//test_genX.batch_size)

    df1 = read_csv('../Aidata/aoi_race/test.csv')
    df2= pd.DataFrame(output)
    df2=df2.idxmax(axis=1)
    new_df=pd.concat([df1,df2],axis=1,join_axes=[df1.index])
    write_csv(new_df)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()