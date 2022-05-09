import pandas as pd
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

# input len for RNN perdiction (48 = 2 days worth of data)
SEQ_LEN = 48
# how far into the future should the model predict (currently 1h)
FUTURE_PERIOD_PREDICT = 1
RATIO_TO_PREDICT = 'DOT'
# model parameters
EPOCHS = 10
BATCH_SIZE = 64
# log outsting to save model progress
NAME = f'{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}'

def classify(current, future):
    # evaluates if the price is perdicted to go up.
    # input: current price and predicted future price
    # output: 1 if predicted to go up, 0 if predicted to go down
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocessdf(df):
    # preprocesses/generates data for RNN model
    # input: dataframe containing all price information for each crypto
    # output: shuffled numpy array containing current price (x) and predicted price (y).
    df = df.drop('Future', 1)
    # iterate through each column in dataframe and scale data
    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    #drop 0 values
    df.dropna(inplace=True)
    sequential_data = []
    #deque takes in a list of a set length, and as another item gets added to list it drops the first item
    prev_days = deque(maxlen=SEQ_LEN)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    #shuffle data
    random.shuffle(sequential_data)

    # generation of Y data if price goes down (sells) if price goes up (buys)
    buys = []
    sells = []
    for seq, target in sequential_data:
        if target ==0:
            sells.append([seq, target])
        elif target ==1:
            buys.append([seq, target])

    # various random shuffling techniques
    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]
    sequential_data = buys + sells
    random.shuffle(sequential_data)

    #generation of final training data in sequential order
    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), np.array(y)

# all price crypto prices (relative to american dollar) used for training
pairs = ['DOTUSD',
        'BTCUSD',
        'ETHUSD',
        'SOLUSD',
        'XRPUSD',
        'LTCUSD',
        'AXSUSD',
        'ZRXUSD']

# all price information for each crypto
main_df = pd.DataFrame()

for pair in pairs:
    # iterates over each crypto price and appends the trading information the main_df
    coinname = pair.split('U')[0]
    df = pd.read_csv(f'{coinname}-price-SEPT1-OCT31.csv', index_col='Date')
    df.drop(['Unnamed: 0', 'Open', 'High', 'Low'], axis=1, inplace=True)
    df.rename(columns={'Close':f'{coinname}_close', 'Volume':f'{coinname}_Volume'}, inplace=True)
    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)


main_df['Future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['Future']))

# set aside 5% of data for validation
times = sorted(main_df.index.values)
last_5pc = times[-int(0.05*len(times))]
validation = main_df[(main_df.index >= last_5pc)]
main_df = main_df[(main_df.index <= last_5pc)]

# generate training and validation set
train_x, train_y = preprocessdf(main_df)
val_x, val_y = preprocessdf(validation)

# RNN model archetecture
model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=0.000001)

# compile the model using the Adam optimizer and categorical crossentropy prediction
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# output model training to tensorboard for realtime model training feedback
tensorboard = TensorBoard(log_dir=f'Logs/{NAME}')

checkpoint_filepath = '/models/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# model training command including the validation data
history = model.fit(train_x,
                    train_y,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(val_x, val_y),
                    callbacks=[tensorboard, model_checkpoint_callback])