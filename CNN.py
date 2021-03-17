import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,BatchNormalization,AveragePooling2D,concatenate,Input,MaxPooling2D,GlobalAveragePooling2D,add

from keras.models import Model

class cnn():

    def AlexNet(self,width, height, depth, classes):
        model = Sequential()
        model.add(
            Conv2D(filters=96, kernel_size=11, strides=(4, 4), input_shape=(width, height, depth), activation='relu',
                   padding='valid'))
        model.add(MaxPool2D(pool_size=3, strides=(2, 2)))
        model.add(Conv2D(filters=256, kernel_size=5, strides=(1, 1), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=3, strides=(2, 2)))
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        print(model.summary())
        return model

    def VGG16Net(self,width, height, depth, classes):
        model = Sequential()

        model.add(
            Conv2D(64, (11, 11), strides=(4, 4), input_shape=(width, height, depth), padding='same', activation='relu'))
        model.add(Conv2D(64, (11, 11), strides=(4, 4), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=3, strides=(2, 2)))
        # model.add(Conv2D(128, (3, 2), strides=(1, 1), padding='same', activation='relu'))
        # model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        # model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=3, strides=(2, 2)))
        # model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        # model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        # model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        # model.add(MaxPool2D(pool_size=(2, 2)))
        # model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        print(model.summary())
        return model

    # Define convolution with batchnromalization
    def Conv2d_BN(self,x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x

    # Define Inception structure
    def Inception(self,x, nb_filter_para):
        (branch1, branch2, branch3, branch4) = nb_filter_para
        branch1x1 = Conv2D(branch1[0], (1, 1), padding='same', strides=(1, 1), name=None)(x)

        branch3x3 = Conv2D(branch2[0], (1, 1), padding='same', strides=(1, 1), name=None)(x)
        branch3x3 = Conv2D(branch2[1], (3, 3), padding='same', strides=(1, 1), name=None)(branch3x3)

        branch5x5 = Conv2D(branch3[0], (1, 1), padding='same', strides=(1, 1), name=None)(x)
        branch5x5 = Conv2D(branch3[1], (1, 1), padding='same', strides=(1, 1), name=None)(branch5x5)

        branchpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branchpool = Conv2D(branch4[0], (1, 1), padding='same', strides=(1, 1), name=None)(branchpool)

        x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)

        return x

    # Build InceptionV1 model
    def InceptionV1(self,width, height, depth, classes):
        input = Input(shape=(width, height, depth))

        x = self.Conv2d_BN(input, 64, (7, 7), strides=(2, 2), padding='same')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = self.Conv2d_BN(x, 192, (3, 3), strides=(1, 1), padding='same')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x = self.Inception(x, [(64,), (96, 128), (16, 32), (32,)])  # Inception 3a 28x28x256
        x = self.Inception(x, [(128,), (128, 192), (32, 96), (64,)])  # Inception 3b 28x28x480
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 14x14x480

        x = self.Inception(x, [(192,), (96, 208), (16, 48), (64,)])  # Inception 4a 14x14x512
        x = self.Inception(x, [(160,), (112, 224), (24, 64), (64,)])  # Inception 4a 14x14x512
        x = self.Inception(x, [(128,), (128, 256), (24, 64), (64,)])  # Inception 4a 14x14x512
        x = self.Inception(x, [(112,), (144, 288), (32, 64), (64,)])  # Inception 4a 14x14x528
        x = self.Inception(x, [(256,), (160, 320), (32, 128), (128,)])  # Inception 4a 14x14x832
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 7x7x832

        x = self.Inception(x, [(256,), (160, 320), (32, 128), (128,)])  # Inception 5a 7x7x832
        x = self.Inception(x, [(384,), (192, 384), (48, 128), (128,)])  # Inception 5b 7x7x1024

        # Using AveragePooling replace flatten
        x = AveragePooling2D(pool_size=(8, 8), strides=(4, 4), padding='same')(x)
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1000, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(classes, activation='softmax')(x)

        model = Model(inputs=input, outputs=x)
        model.summary()
        return model

    def Residual_Block(self,input_model, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
        x = self.Conv2d_BN(input_model, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = self.Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')

        # need convolution on shortcut for add different channel
        if with_conv_shortcut:
            shortcut = self.Conv2d_BN(input_model, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, input_model])
            return x

    # Built ResNet34
    def ResNet34(self,width, height, depth, classes):
        Img = Input(shape=(width, height, depth))

        x = self.Conv2d_BN(Img, 64, (7, 7), strides=(2, 2), padding='same')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        # Residual conv2_x ouput 56x56x64
        x = self.Residual_Block(x, nb_filter=64, kernel_size=(3, 3))

        # Residual conv3_x ouput 28x28x128
        x = self.Residual_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2),
                           with_conv_shortcut=True)  # need do convolution to add different channel
        x = self.Residual_Block(x, nb_filter=128, kernel_size=(3, 3))

        # Residual conv4_x ouput 14x14x256
        x = self.Residual_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2),
                           with_conv_shortcut=True)  # need do convolution to add different channel
        x = self.Residual_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = self.Residual_Block(x, nb_filter=256, kernel_size=(3, 3))

        # Residual conv5_x ouput 7x7x512
        x = self.Residual_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)

        x = self.Residual_Block(x, nb_filter=512, kernel_size=(3, 3))

        # Using AveragePooling replace flatten
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, activation='softmax')(x)

        model = Model(inputs=Img, outputs=x)
        return model