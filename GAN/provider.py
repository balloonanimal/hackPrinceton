import numpy as np
import tensorflow as tf

class provider():
    def __init__(self):
        self.test_labels, self.test_data, self.train_labels, self.train_data = self.load_emnist()
        self.counter = 0
        self.test_size = len(self.test_labels)
        self.train_size = len(self.train_labels)
        self.num_classes = 26

    def load_emnist(self):
        print('loading test data')
        with open("test.csv", "r") as test_file:
            test_matrix = np.loadtxt(test_file, delimiter=',')
            test_labels = test_matrix[:, 0].astype(np.uint8)
            test_data = test_matrix[:, 1:].astype(np.uint8)

        print('loading training data')        
        with open("train.csv", "r") as train_file:
            train_matrix = np.loadtxt(train_file, delimiter=',')
            train_labels = train_matrix[:, 0].astype(np.uint8)
            train_data = train_matrix[:, 1:].astype(np.uint8)

        def filter_labels(labels, data):
            relevant_labels = list(range(10, 36))
            mask = np.isin(labels, relevant_labels)
            return labels[mask], data[mask]

        def reshape_data(data):
            l = len(data)
            d =  data.reshape((l, 28, 28))
            return d

        filtered_test_labels, filtered_test_data = filter_labels(test_labels, test_data)
        filtered_train_labels, filtered_train_data = filter_labels(train_labels, train_data)

        def quantize_labels(labels):
            return labels - 10

        return quantize_labels(filtered_test_labels), reshape_data(filtered_test_data), quantize_labels(filtered_train_labels), reshape_data(filtered_train_data)

    def provide_data(self, split_name, batch_size):
        if split_name == 'test':
            labels = self.test_labels
            data = self.test_data
            split_size = self.test_size
        elif split_name == 'train':
            labels = self.train_labels
            data = self.train_data
            split_size = self.train_size
        else:
            raise ValueError('invalid split name')
        
        counter = self.counter
        label_batch = labels.take(range(counter, counter + batch_size), axis=0, mode='wrap')
        data_batch = data.take(range(counter, counter + batch_size), axis=0, mode='wrap')        
        self.counter = (counter + batch_size) % split_size

        labels = tf.convert_to_tensor(label_batch)
        images = tf.convert_to_tensor(np.expand_dims(data_batch, axis=3))
        
        images = (tf.to_float(images) - 128.0) / 128.0

        one_hot_labels = tf.one_hot(labels, self.num_classes)
        return images, one_hot_labels
