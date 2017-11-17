import cv2
import pandas
import numpy
import pickle

from matplotlib import pyplot

from time import gmtime, strftime

from sklearn.metrics import confusion_matrix, accuracy_score

from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Conv2D, Dense, MaxPool2D, Input, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from keras.preprocessing.image import ImageDataGenerator


EPOCHS = 50
BATCH_SIZE = 256

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
    model = Sequential()
    input_shape = (48, 48, 1)

    # 48*48*1
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    # 24*24*64
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    # 12*12*128
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    # 6*6*256
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    # 3*3*512
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3)))

    # flatten: 6*6*256
    model.add(Flatten())

    # 1*(6*6*256)
    model.add(Dense(1014, activation='relu'))

    # 1*1024
    model.add(Dense(7, activation='softmax'))

    return model

def plotHistory(timestamp, history):
    fig = pyplot.figure(figsize=(8, 8))
    pyplot.subplot(211)
    pyplot.plot(history.history["loss"])
    pyplot.plot(history.history["val_loss"])
    pyplot.title("Training & testing loss curve")
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.legend(["train", "test"], loc='upper right')

    pyplot.subplot(212)
    pyplot.plot(history.history["acc"])
    pyplot.plot(history.history["val_acc"])
    pyplot.title("Training & testing acc curve")
    pyplot.xlabel("epoch")
    pyplot.ylabel("acc")
    pyplot.legend(["train", "test"], loc='upper left')

    fig.savefig(timestamp+".png" )

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

    total_num = len(x)
    batch_size= 64

    model = constructModel()
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    print "fitting..."
    # checkpoint

    timestamp = strftime("%Y-%m-%d-%H%M%S", gmtime())

    filepath="log/%s.weights.{epoch:02d}-{val_acc:.2f}.hdf5"%timestamp

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc')
    early_stop = EarlyStopping(monitor='val_acc', patience=5, mode='max')
    logger = CSVLogger("log/%s.logger.csv"%timestamp, separator=',', append=False)

    callbacks_list = [checkpoint, early_stop, logger]

    datagen = ImageDataGenerator(horizontal_flip=True)
    x_y = datagen.flow(x, y, batch_size=batch_size)

    # history = model.fit_generator(x_y, steps_per_epoch=2*total_num/batch_size, epochs=100, callbacks=callbacks_list, validation_data=(x_test, y_test))

    history = model.fit(x, y, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list)
    # plotHistory(timestamp, history)

    test_pred = model.predict(x_test)

    true_label = numpy.argmax(y_test, axis=1)
    pred_label = numpy.argmax(test_pred, axis=1)

    print accuracy_score(true_label, pred_label)
    print confusion_matrix(true_label, pred_label)







