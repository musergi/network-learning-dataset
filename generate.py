import os
import argparse
import itertools
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd


class RecorderCallback(tf.keras.callbacks.Callback):
    def __init__(self, recording_dir='.'):
        super(RecorderCallback, self).__init__()
        self.recording_dir = recording_dir
        self.config = {}
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['epoch'] = epoch
        logs.update(self.config)
        self.records.append(logs)
        print(logs)

    def save(self):
        filepath = os.path.join(self.recording_dir, 'index.csv')
        pd.DataFrame(self.records).to_csv(filepath)


def generate_configs(layer_sizes, layer_counts, iterations):
    network_id = 1
    for iteration in range(iterations):
        for layer_count in layer_counts:
            layer_possible_sizes = [layer_sizes] * layer_count
            for combination in itertools.product(*layer_possible_sizes):
                out_dict = {'iteration': iteration, 'layer_count': layer_count,
                    'network_id': network_id} 
                for index, layer in enumerate(combination):
                    out_dict[f'layer{index}'] = layer
                yield out_dict
                network_id += 1


def create_model(config):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    for index in range(config['layer_count']):
        layer_size = config[f'layer{index}'] 
        model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_size_start', type=int, required=True)
    parser.add_argument('--layer_size_stop', type=int, required=True)
    parser.add_argument('--layer_size_step', type=int, required=True)
    parser.add_argument('--layer_count_start', type=int, required=True)
    parser.add_argument('--layer_count_stop', type=int, required=True)
    parser.add_argument('--layer_count_step', type=int, required=True)
    parser.add_argument('--iterations', type=int, default=1)
    args = parser.parse_args()
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    layer_sizes = list(range(args.layer_size_start, args.layer_size_stop,
                args.layer_size_step))
    layer_counts = list(range(args.layer_count_start, args.layer_count_stop,
                args.layer_count_step))
    recorder = RecorderCallback()
    for config in generate_configs(layer_sizes, layer_counts, args.iterations):
        model = create_model(config)
        recorder.config = config
        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                epochs=10, verbose=0, callbacks=[recorder])
        print(f'Trained config={config}')
    recorder.save()
    return 0


if __name__ == '__main__':
    exit(main())
