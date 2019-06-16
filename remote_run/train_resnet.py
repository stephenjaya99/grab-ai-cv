#!/usr/bin/env python
# coding: utf-8

from load_data import create_generators
from model_resnet50 import get_resnet_model

train_generator, validation_generator, test_generator = create_generators()

model = get_resnet_model()

epochs = 10
batch_size = 32
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

history = model.fit_generator(train_generator,
                              epochs=epochs,
                              workers=4,
                              validation_data=validation_generator
                              )
