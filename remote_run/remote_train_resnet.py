import os
import matplotlib.pyplot as plt
import tensorflow as tf
from load_data import create_generators
from model_resnet50 import get_resnet_model

from azureml.core import Run
print("TensorFlow version:", tf.VERSION)

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str,
                    dest='data_folder', help='data folder mounting point')
parser.add_argument('--batch-size', type=int, dest='batch_size',
                    default=32, help='mini batch size for training')
parser.add_argument('--epoch', type=int, dest='epoch',
                    default=10, help='epoch')
args = parser.parse_args()

data_folder = os.path.join(args.data_folder, 'cars')

print('training dataset is stored here:', data_folder)

train_generator, validation_generator, test_generator = create_generators(
    data_folder)

model = get_resnet_model()

epochs = args.epoch
batch_size = args.batch_size
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

# start an Azure ML run
run = Run.get_context()


class LogRunMetrics(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log):
        run.log('Loss', log['loss'])
        run.log('Accuracy', log['acc'])


history = model.fit_generator(
    train_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    workers=4,
    verbose=2,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[LogRunMetrics()]
)

score = model.evaluate_generator(
    validation_generator,
    steps=validation_steps,
    workers=4,
    verbose=0
)

# log a single value
run.log("Final test loss", score[0])
print('Test loss:', score[0])

run.log('Final test accuracy', score[1])
print('Test accuracy:', score[1])

plt.figure(figsize=(6, 3))
plt.title('Stanford Cars with Keras ({} epochs)'.format(n_epochs), fontsize=14)
plt.plot(history.history['acc'], 'b-', label='Accuracy', lw=4, alpha=0.5)
plt.plot(history.history['loss'], 'r--', label='Loss', lw=4, alpha=0.5)
plt.legend(fontsize=12)
plt.grid(True)

# log an image
run.log_image('Accuracy vs Loss', plot=plt)

# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)

# serialize NN architecture to JSON
model_json = model.to_json()
# save model JSON
with open('./outputs/model/model.json', 'w') as f:
    f.write(model_json)
# save model weights
model.save_weights('./outputs/model/model_resnet50.h5')
print("model saved in ./outputs/model folder")
