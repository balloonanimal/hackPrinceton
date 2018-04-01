import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tfgan = tf.contrib.gan

class Converter():
    def __init__(self):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        saver = tf.train.import_meta_graph('./haiku/model/model.ckpt.meta')
        saver.restore(sess, './haiku/model/model.ckpt')
        graph = tf.get_default_graph()

        one_hot = graph.get_tensor_by_name('one_hot_1:0')
        noise = graph.get_tensor_by_name('random_normal_1:0')
        output = graph.get_tensor_by_name('Generator_1/Conv/Tanh:0')

        self.sess, self.one_hot, self.noise, self.output = sess, one_hot, noise, output

        self.mapping = {'A': 0,
                        'B': 1,
                        'C': 2,
                        'D': 3,
                        'E': 4,
                        'F': 5,
                        'G': 6,
                        'H': 7,
                        'I': 8,
                        'J': 9,
                        'K': 10,
                        'L': 11,
                        'M': 12,
                        'N': 13,
                        'O': 14,
                        'P': 15,
                        'Q': 16,
                        'R': 17,
                        'S': 18,
                        'T': 19,
                        'U': 20,
                        'V': 21,
                        'W': 22,
                        'X': 23,
                        'Y': 24,
                        'Z': 25}

    def convert(self, letters):
        l = len(letters)
        new_one_hot = np.zeros((500, 26))
        for index, letter in enumerate(letters):
            if letter != ' ':
                i = self.mapping[letter]
                new_one_hot[index][i] = 1

        new_noise = np.random.normal(size=(500, 64))

        feed_dict = {self.one_hot: new_one_hot, self.noise: new_noise}

        o = self.sess.run(self.output, feed_dict)

        denormed = o * 128.0 - 128.0

        denormed = np.squeeze(denormed)

        for index, letter in enumerate(letters):
            if letter == ' ':
                denormed[index] = np.zeros((28, 28))

        int_images = denormed.astype(np.uint8)

        return int_images[:l]
