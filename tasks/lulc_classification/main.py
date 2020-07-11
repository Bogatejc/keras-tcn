import numpy as np
import os
from pathlib import Path

import keras
from keras import layers
from keras import backend as K
from keras.utils import to_categorical
from tcn import compiled_tcn
from sklearn.model_selection import train_test_split

def data_generator():
    DATA = np.load('../../testni_input_file.npy')
    GT_DATA = np.load('../../LPIS_ground_truth.npy')
    GT_DATA_binary = to_categorical(GT_DATA)

    for i in range(0, DATA.shape[0]):
      for j in range(0, DATA.shape[1]):
        for k in range(0, DATA.shape[2]):
          if(np.isnan(DATA[i][j][k])):
            DATA[i][j][k] = 0
        
    X_train, X_test, y_train, y_test = train_test_split(DATA, GT_DATA_binary, test_size=0.30, shuffle=True)

    return (X_train, y_train), (X_test, y_test)

def run_task():
    (x_train, y_train), (x_test, y_test) = data_generator()
    print('Data prepared')

    model = compiled_tcn(num_feat=7,
                         num_classes=26,
                         nb_filters=128,
                         kernel_size=3,
                         dilations=[2 ** i for i in range(3)],
                         nb_stacks=1,
                         lr=0.001, #5e-4,
                         max_len=34,
                         use_skip_connections=False,
                         return_sequences=False)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')
    print(f'x_test.shape = {x_test.shape}')
    print(f'y_test.shape = {y_test.shape}')

    model.summary()

    model.compile('adam', 'categorical_crossentropy', metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=2, batch_size=2, validation_data=(x_test, y_test))
    del model

if __name__ == '__main__':
    run_task()