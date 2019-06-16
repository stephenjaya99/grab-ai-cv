import os
import pandas as pd
from scipy.io import loadmat
import tensorflow as tf


def load_dataframe(data_folder):
    train_dir = os.path.join(data_folder, 'cars_train')
    test_dir = os.path.join(data_folder, 'cars_test')
    devkir_dir = os.path.join(data_folder, 'devkit')

    cars_meta = loadmat(os.path.join(data_folder, devkir_dir, 'cars_meta.mat'))
    cars_train_annos = loadmat(os.path.join(
        data_folder, devkir_dir, 'cars_train_annos.mat'))
    cars_test_annos = loadmat(os.path.join(
        data_folder, devkir_dir, 'cars_test_annos.mat'))

    cars_meta = loadmat(os.path.join(devkir_dir, 'cars_meta.mat'))
    labels = [c for c in cars_meta['class_names'][0]]
    labels = pd.DataFrame(labels, columns=['labels'])

    train_columns = ['class', 'fname']
    frame = [[i.flat[0] for i in [line[4], line[5]]]
             for line in cars_train_annos['annotations'][0]]
    df_train = pd.DataFrame(frame, columns=train_columns)
    # Python indexing starts on zero.
    df_train['class'] = df_train['class'] - 1
    df_train = df_train.merge(labels, left_on='class', right_index=True)
    df_train = df_train.sort_index()

    test_columns = ['fname']
    frame = [[i.flat[0] for i in [line[4]]]
             for line in cars_test_annos['annotations'][0]]
    df_test = pd.DataFrame(frame, columns=test_columns)

    return df_train, df_test


def create_generators(data_folder):
    train_dir = os.path.join(data_folder, 'cars_train')
    test_dir = os.path.join(data_folder, 'cars_test')

    df_train, df_test = load_dataframe(data_folder)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.2,
        rotation_range=20,
        horizontal_flip=True,
    )
    train_generator = datagen.flow_from_dataframe(
        df_train,
        directory=os.path.join('./', train_dir),
        x_col='fname',
        y_col='class',
        target_size=(224, 224),
        subset='training',
        class_mode="other",
    )
    validation_generator = datagen.flow_from_dataframe(
        df_train,
        directory=os.path.join('./', train_dir),
        x_col='fname',
        y_col='class',
        target_size=(224, 224),
        subset='validation',
        class_mode="other",
    )
    test_generator = datagen.flow_from_dataframe(
        df_test,
        directory=os.path.join('./', test_dir),
        x_col='fname',
        target_size=(224, 224),
        class_mode=None,
    )

    return train_generator, validation_generator, test_generator
