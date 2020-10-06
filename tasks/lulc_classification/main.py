import numpy as np
import os
from pathlib import Path
import re

import keras
from keras import layers
from keras import backend as K
from keras.utils import to_categorical
from tcn import compiled_tcn
from sklearn.metrics import confusion_matrix, classification_report

def prepare_input_data():
  # DROP CLASSES: [1,3,4,5,8,14,16,18,19,21]
    train_files = ['train_class_2.npy', 'train_class_6.npy', 'train_class_7.npy', 'train_class_9.npy', 'train_class_10.npy', 'train_class_11.npy', 
                  'train_class_12.npy', 'train_class_13.npy', 'train_class_15.npy', 'train_class_17.npy', 'train_class_20.npy',
                  'train_class_22.npy', 'train_class_23.npy', 'train_class_24.npy', 'train_class_25.npy']
    test_files = ['test_class_2.npy', 'test_class_6.npy', 'test_class_7.npy', 'test_class_9.npy','test_class_10.npy', 'test_class_11.npy',
                  'test_class_12.npy', 'test_class_13.npy', 'test_class_15.npy', 'test_class_17.npy', 'test_class_20.npy',
                  'test_class_22.npy', 'test_class_23.npy', 'test_class_24.npy', 'test_class_25.npy']


    train_input = []
    for f in train_files:
        print(f)
        tmp_file = np.load('./Inputs/Train/' + f)

        gt_num = re.findall(r'\d+', f)

        for i in range(0, tmp_file.shape[0]):
            
            for j in range(0, 34):
                for k in range(0, 7):
                    if(np.isnan(tmp_file[i][j][k])):
                      tmp_file[i][j][k] = 0  

            train_input.append(tmp_file[i])

    test_input = []
    for f in test_files:
        tmp_file = np.load('./Inputs/Test/' + f)

        gt_num = re.findall(r'\d+', f)

        for i in range(0, tmp_file.shape[0]):

            for j in range(0, 34):
                for k in range(0, 7):
                    if(np.isnan(tmp_file[i][j][k])):
                      tmp_file[i][j][k] = 0  
                      
            test_input.append(tmp_file[i])
    
    np.save('./Inputs/train_input_distribucija.npy', train_input)
    np.save('./Inputs/test_input_distribucija.npy', test_input)

def prepare_gt_data_ones():
    # DROP CLASSES: [1,3,4,5,8,14,16,18,19,21]
    train_files = ['train_gt_ones_2.npy', 'train_gt_ones_6.npy', 'train_gt_ones_7.npy', 'train_gt_ones_9.npy', 'train_gt_ones_10.npy', 'train_gt_ones_11.npy', 
                'train_gt_ones_12.npy', 'train_gt_ones_13.npy', 'train_gt_ones_15.npy', 'train_gt_ones_17.npy', 'train_gt_ones_20.npy',
                'train_gt_ones_22.npy', 'train_gt_ones_23.npy', 'train_gt_ones_24.npy', 'train_gt_ones_25.npy']
    test_files = ['test_gt_ones_2.npy', 'test_gt_ones_6.npy', 'test_gt_ones_7.npy', 'test_gt_ones_9.npy','test_gt_ones_10.npy', 'test_gt_ones_11.npy',
                'test_gt_ones_12.npy', 'test_gt_ones_13.npy', 'test_gt_ones_15.npy', 'test_gt_ones_17.npy', 'test_gt_ones_20.npy',
                'test_gt_ones_22.npy', 'test_gt_ones_23.npy', 'test_gt_ones_24.npy', 'test_gt_ones_25.npy']


    train_output_ones = []
    for f in train_files:
        print(f)
        tmp_file = np.load('./Inputs/Train/' + f)

        for i in range(0, tmp_file.shape[0]):
            train_output_ones.append(tmp_file[i])


    test_output_ones = []
    for f in test_files:
        print(f)
        tmp_file = np.load('./Inputs/Test/' + f)

        for i in range(0, tmp_file.shape[0]):
            test_output_ones.append(tmp_file[i])
    
    np.save('./Inputs/train_output_ones_distribucija.npy', train_output_ones)
    np.save('./Inputs/test_output_distribucija.npy', test_output_ones)

def prepare_gt_data_percentage():
    # DROP CLASSES: [1,3,4,5,8,14,16,18,19,21]
    train_files = ['train_gt_multi_2.npy', 'train_gt_multi_6.npy', 'train_gt_multi_7.npy', 'train_gt_multi_9.npy', 'train_gt_multi_10.npy', 'train_gt_multi_11.npy', 
                'train_gt_multi_12.npy', 'train_gt_multi_13.npy', 'train_gt_multi_15.npy', 'train_gt_multi_17.npy', 'train_gt_multi_20.npy',
                'train_gt_multi_22.npy', 'train_gt_multi_23.npy', 'train_gt_multi_24.npy', 'train_gt_multi_25.npy']
    test_files = ['test_gt_multi_2.npy', 'test_gt_multi_6.npy', 'test_gt_multi_7.npy', 'test_gt_multi_9.npy','test_gt_multi_10.npy', 'test_gt_multi_11.npy',
                'test_gt_multi_12.npy', 'test_gt_multi_13.npy', 'test_gt_multi_15.npy', 'test_gt_multi_17.npy', 'test_gt_multi_20.npy',
                'test_gt_multi_22.npy', 'test_gt_multi_23.npy', 'test_gt_multi_24.npy', 'test_gt_multi_25.npy']


    train_output_percentage = []
    for f in train_files:
        print(f)
        tmp_file = np.load('./Inputs/Train/' + f)

        for i in range(0, tmp_file.shape[0]):
            train_output_percentage.append(tmp_file[i])


    test_output_percentage = []
    for f in test_files:
        print(f)
        tmp_file = np.load('./Inputs/Test/' + f)

        for i in range(0, tmp_file.shape[0]):
            test_output_percentage.append(tmp_file[i])
    
    np.save('./Inputs/train_output_percentage_distribucija.npy', train_output_percentage)
    np.save('./Inputs/test_output_percentage_distribucija.npy', test_output_percentage)

def run_task():
    prepare_input_data()
    prepare_gt_data_ones()
    prepare_gt_data_percentage()
    X_train = np.load('./Inputs/train_input_distribucija.npy')
    X_test = np.load('./Inputs/test_input_distribucija.npy')
    y_train = np.load('./Inputs/train_output_ones_distribucija.npy')
    y_test = np.load('./Inputs/test_output_distribucija.npy')
    # y_train = np.load('./Inputs/train_output_percentage_distribucija.npy')
    # y_test = np.load('./Inputs/test_output_percentage_distribucija.npy')
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

    print(f'x_train.shape = {X_train.shape}')
    print(f'y_train.shape = {y_train.shape}')
    print(f'x_test.shape = {X_test.shape}')
    print(f'y_test.shape = {y_test.shape}')

    model.summary()

    model.compile('adam', 'categorical_crossentropy', metrics=["accuracy"])

    history = model.fit(X_train, y_train, epochs=5, batch_size=2, validation_data=(X_test, y_test))_train, y_train, epochs=2, batch_size=2, validation_data=(X_test, y_test))
    predictions = model.predict(X_test)
    
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(predictions = model.predict(X_test), axis=1)))

    model.save('./Models')

    del model

if __name__ == '__main__':
    run_task()