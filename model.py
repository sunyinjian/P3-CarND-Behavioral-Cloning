import helper

from keras.models import  Sequential
from keras.layers import Cropping2D, Lambda, Conv2D, Flatten, Dense, Dropout, MaxPooling2D, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization

EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

#The model is based on the Nvidia's paper "End to End Learning for Self-Driving Cars"
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode='same', init='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode='same', init='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode='same', init='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Conv2D(64, 3, 3, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, 3, 3, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(100, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(50, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(10, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(1, init='uniform'))
model.summary()

model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

train_samples, validation_samples = helper.load_csv_records(['./track1_data/driving_log.csv', './track2_data/driving_log.csv'])

samples_per_epoch = (20000//BATCH_SIZE)*BATCH_SIZE
nb_val_samples = 4000

train_generator = helper.generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = helper.generator(validation_samples, batch_size=BATCH_SIZE)

checkpoint = ModelCheckpoint('./model/model{epoch:02d}--loss{val_loss:.4f}.h5', monitor='val_loss', save_best_only=True)

history_object = model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, validation_data=validation_generator, nb_val_samples=nb_val_samples, nb_epoch=EPOCHS, callbacks=[checkpoint], verbose=1)

import  matplotlib.pyplot as plt
print(history_object.history.keys())# print the keys contained in the history object
plt.plot(history_object.history['loss']) ## plot the training and validation loss for each epoch
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('./loss.png')

model.save('model.h5')
