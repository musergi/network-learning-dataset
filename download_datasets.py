import os
import pickle
import tensorflow as tf

os.makedirs('dataset', exist_ok=True)

# MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
with open(os.path.join('dataset', 'mnist.pickle'), 'wb') as f:
    dataset = (x_train, y_train), (x_test, y_test)
    pickle.dump(dataset, f)

