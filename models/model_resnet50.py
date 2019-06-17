#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf


def get_resnet_model():
    base_model = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=False,
    )
    for layer in base_model.layers:
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(196, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.SGD(lr=0.005, decay=0.0001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])

    return model
