import numpy as np
import os
from pathlib import Path

from tcn import compiled_tcn
from sklearn.model_selection import train_test_split

def data_generator():
    DATA = np.load('../../testni_input_file.npy')
    GT_DATA = np.load('../../LPIS_ground_truth.npy')
        
    X_train, X_test, y_train, y_test = train_test_split(DATA, GT_DATA, test_size=0.30, shuffle=True)
    return (X_train, y_train), (X_test, y_test)

def run_task():
    (x_train, y_train), (x_test, y_test) = data_generator()

    model = compiled_tcn(return_sequences=False,
                         num_feat=7,
                         num_classes=26,
                         nb_filters=128,
                         kernel_size=3,
                         dilations=[2 ** i for i in range(5)],
                         nb_stacks=1,
                         max_len=34,
                         use_skip_connections=True)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')
    print(f'x_test.shape = {x_test.shape}')
    print(f'y_test.shape = {y_test.shape}')

    model.summary()

    model.fit(x_train, y_train, epochs=2, batch_size=128, validation_data=(x_test, y_test))

    test_acc = model.evaluate(x=x_test, y=y_test)[1]
    print('Test accuracy: ', test_acc)

if __name__ == '__main__':
    run_task()