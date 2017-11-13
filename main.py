import cv2
import pandas
import numpy

from matplotlib import pyplot

from keras.models import Model
from keras.utils import to_categorical
from keras.layers import Conv2D, Dense, MaxPool2D, Input, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping


def loadData():
    print "loading data..."
    raw_data = pandas.read_csv("data/fer2013.csv")

    raw_data['image'] = raw_data['pixels'].apply(
        lambda x: numpy.fromstring(x, dtype=numpy.uint8, sep=' ').reshape((48, 48, 1))
    )

    cols = ['emotion', 'image']

    training = raw_data[raw_data[u'Usage'] == u"Training"].reset_index(drop=True)
    public_test = raw_data[raw_data[u'Usage'] == u"PublicTest"].reset_index(drop=True)
    private_test = raw_data[raw_data[u'Usage'] == u"PrivateTest"].reset_index(drop=True)

    training = training[cols]
    public_test = public_test[cols]
    private_test = private_test[cols]

    return training, public_test, private_test

def constructModel():
    print "constructing model..."
    image = Input(shape=(48, 48, 1))

    # 48*48*1
    conv_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(image)
    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv_1)
    pool_1 = MaxPool2D(pool_size=(2, 2))(conv_2)

    # 24*24*64
    conv_3 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2))(conv_3)

    # 12*12*128
    conv_4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(pool_2)
    pool_3 = MaxPool2D(pool_size=(2, 2))(conv_4)

    # 6*6*256
    conv_5 = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(pool_3)
    pool_4 = MaxPool2D(pool_size=(2, 2))(conv_5)

    # 3*3*512
    conv_6 = Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu')(pool_4)
    pool_5 = MaxPool2D(pool_size=(3, 3))(conv_6)

    # flatten
    flat = Flatten()(pool_5)

    # 1*256
    dense_1 = Dense(7, activation='softmax')(flat)

    model = Model(inputs=image, outputs=dense_1)
    return model

def plotHistory(history):
    pyplot.figure()
    pyplot.plot(history.history["loss"])
    pyplot.plot(history.history["val_loss"])
    pyplot.title("Training & testing loss curve")
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.legend(["train", "test"], loc='upper right')

    pyplot.figure()
    pyplot.plot(history.history["coeff_determination"])
    pyplot.plot(history.history["val_coeff_determination"])
    pyplot.title("Training & testing R2 score curve")
    pyplot.xlabel("epoch")
    pyplot.ylabel("R2 score")
    pyplot.legend(["train", "test"], loc='upper left')

def dataFormatting(data):
    print "formatting..."
    temp_x = []
    temp_y = []

    for index, row in data.iterrows():
        temp_img = numpy.empty_like(row[u'image'], dtype=numpy.float32)
        cv2.normalize(row[u'image'], temp_img, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        temp_x.append(temp_img)
        temp_y.append(row[u'emotion'])

    x = numpy.array(temp_x)
    y = to_categorical(numpy.array(temp_y), 7)

    return x, y

if __name__ == "__main__":
    training, public_test, private_test = loadData()

    x, y = dataFormatting(training)
    x_test, y_test = dataFormatting(public_test)

    model = constructModel()
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    print "fitting..."
    # checkpoint

    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    early_stop = EarlyStopping(monitor='val_acc', patience=5, mode='auto')

    callbacks_list = [checkpoint, early_stop]

    history = model.fit(x, y, validation_data=(x_test, y_test), epochs=100, callbacks=callbacks_list)
    plotHistory(history)

