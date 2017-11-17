import os
import cv2
import pandas
import numpy
import pickle

from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.utils import to_categorical, plot_model

from keras.layers import Conv2D, Dense, MaxPool2D, Input, \
    Flatten, Dropout, BatchNormalization, Activation

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from keras.preprocessing.image import ImageDataGenerator

EPOCHS = 50
BATCH_SIZE = 256
TRAINING_TESTING_RATIO = 0.7

PATIENCE_EPOCHS = 10

class Logger():
    def __init__(self):
        self.loss_measure = "loss"
        self.acc_measure = "acc"

        from time import gmtime, strftime
        timestamp = strftime("%Y%m%d%H%M%S", gmtime())
        self.target_path = "log/" + timestamp

        if not os.path.exists(self.target_path):
            os.mkdir(self.target_path)

        import signal
        signal.signal(signal.SIGINT, self.signalHandler)

    def signalHandler(self, signal, frame):
        print "system signal (interrupt or kill) received"
        import shutil
        shutil.rmtree(self.target_path)
        import sys
        sys.exit(1)

    def getLoggerName(self):
        return self.target_path + "/logger.csv"

    def getWeightName(self):
        return self.target_path + "/weights.last.hdf5"

    def saveHistory(self, history):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot
        fig = pyplot.figure(figsize=(8, 10))
        pyplot.subplot(211)
        pyplot.plot(history.history[self.loss_measure])
        pyplot.plot(history.history["val_"+self.loss_measure])
        pyplot.title("Training & testing loss curve")
        pyplot.xlabel("epoch")
        pyplot.ylabel("loss")
        pyplot.legend(["train", "test"], loc='upper right')

        pyplot.subplot(212)
        pyplot.plot(history.history[self.acc_measure])
        pyplot.plot(history.history["val_"+self.acc_measure])
        pyplot.title("Training & testing acc curve")
        pyplot.xlabel("epoch")
        pyplot.ylabel("acc")
        pyplot.legend(["train", "test"], loc='upper left')

        fig.savefig(self.target_path + "/history.png" )

    def saveModelArch(self, model):
        from keras.utils import plot_model
        plot_model(model, to_file=self.target_path + "/model.png", show_shapes=True)

    def saveResults(self, true, pred):
        from sklearn.metrics import confusion_matrix, accuracy_score
        conf_matrix = confusion_matrix(true, pred)
        classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        with open(self.target_path+"/mega_info.txt", "w") as fout:
            fout.writelines("Training: epoch {}, batch size {}\n".format(EPOCHS, BATCH_SIZE))
            fout.writelines("Confusion matrix:\n")
            fout.write('{:^10}'.format(' '))
            for each in classes:
                fout.write('{:^10}'.format(each))
            fout.write('\n')
            for i in range(len(classes)):
                fout.write('{:^10}'.format(classes[i]))
                for item in conf_matrix[i]:
                    fout.write('{:^10}'.format(item))
                fout.write('\n')
            fout.write('\n')


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
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))

    # 24*24*64
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))

    # 12*12*128
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=(2, 2)))

    # flatten: 6*6*256
    model.add(Flatten())

    # 1*(6*6*256)
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 1*512
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 1*256
    model.add(Dense(7, activation='softmax'))

    return model

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

def randomSplit(x, y):
    data_size = x.shape[0]
    ratio = TRAINING_TESTING_RATIO
    numpy.random.seed(seed=1)
    print "random splitting, training ratio: %.2f with fixed seed"%ratio

    permu = numpy.random.permutation(data_size)
    select_bool = permu <= (ratio*data_size)

    return x[select_bool], y[select_bool], x[~select_bool], y[~select_bool]

if __name__ == "__main__":
    training, public_test, private_test = loadData()

    x, y = dataFormatting(training)
    x, y, x_val, y_val = randomSplit(x, y)
    x_test, y_test = dataFormatting(public_test)

    total_num = len(x)

    model = constructModel()

    model.compile(loss='categorical_crossentropy',
                  optimizer='SGD',
                  metrics=['accuracy'])

    print "fitting..."
    # checkpoint

    logging = Logger()

    checkpoint = ModelCheckpoint(filepath=logging.getWeightName(),
                                 monitor='val_acc')

    early_stop = EarlyStopping(monitor='val_acc',
                               patience=PATIENCE_EPOCHS,
                               mode='max')

    logger = CSVLogger(filename= logging.getLoggerName(),
                       separator=',',
                       append=False)

    callbacks_list = [checkpoint, early_stop, logger]

    datagen = ImageDataGenerator(horizontal_flip=True)

    x_y = datagen.flow(x=x,
                       y=y,
                       batch_size=BATCH_SIZE)

    # history = model.fit_generator(x_y, steps_per_epoch=2*total_num/batch_size, epochs=100, callbacks=callbacks_list, validation_data=(x_test, y_test))

    history = model.fit(x=x,
                        y=y,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=callbacks_list,
                        validation_data=(x_val, y_val))

    logging.saveHistory(history)
    logging.saveModelArch(model)

    test_pred = model.predict(x_val)

    true_label = numpy.argmax(y_val, axis=1)
    pred_label = numpy.argmax(test_pred, axis=1)

    logging.saveResults(true_label, pred_label)

