from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from load_data import load_dataframe
from model_resnet50 import get_resnet_model

print("TensorFlow version:", tf.VERSION)

data_folder = os.path.join('./', 'data')

model = get_resnet_model()
model.load_weights("./outputs/model_resnet50.h5")

_, df_test = load_dataframe()

cars_meta = loadmat(os.path.join(devkir_dir, 'cars_meta.mat'))
labels = [c for c in cars_meta['class_names'][0]]
labels = pd.DataFrame(labels, columns=['labels'])

with open('./test_result_with_confidence.txt', 'w+') as f:
    for index, row in df_test.iterrows():
        img = Image.open(os.path.join(test_dir, row['fname']), 'r')
        img_resize = img.resize((224, 224))
        img_resize = img_resize.convert('RGB')
        print(row['fname'], np.array(img_resize).shape)
        X_test = np.expand_dims(np.array(img_resize), axis=0)
        y_pred = model.predict(X_test)
        y = np.argmax(y_pred[0], axis=-1)  # multiple categories
        print(y, y_pred[0][y])
        f.write("%s,%s\n" % (y, y_pred[0][y]))
