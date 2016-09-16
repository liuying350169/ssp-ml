# mnistdnn.py

import tensorflow as tf
import numpy as np
from parameterservermodel import ParameterServerModel

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# 2 layers, 1024, 512 neurons

class MnistDNN(ParameterServerModel):
    def __init__(self, batch_size, gpu=False):
	session = tf.Session()

        self.gpu = gpu
        #self.batch_size = batch_size
        input_units = 784
        output_units = 10
        hidden_units = 100  #700

        x = tf.placeholder('float32', shape=[None, input_units], name='x')
        true_y = tf.placeholder('float32', shape=[None, output_units], name='y')

        W_fc0 = weight_variable([input_units, hidden_units], 'W_fc0')
        b_fc0 = bias_variable([hidden_units], 'b_fc0')
        h_fc0 = tf.nn.relu(tf.matmul(x, W_fc0) + b_fc0)

        W_fc1 = weight_variable([hidden_units, output_units], 'W_fc1')
        b_fc1 = bias_variable([output_units], 'b_fc1')

        guess_y = tf.matmul(h_fc0, W_fc1) + b_fc1

        variables = [W_fc0, b_fc0, W_fc1, b_fc1]
        loss = tf.nn.softmax_cross_entropy_with_logits(guess_y, true_y)

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        compute_gradients = optimizer.compute_gradients(loss, variables)
        apply_gradients = optimizer.apply_gradients(compute_gradients)
        minimize = optimizer.minimize(loss)
        correct_prediction = tf.equal(tf.argmax(guess_y, 1), tf.argmax(true_y, 1))
        error_rate = 1 - tf.reduce_mean(tf.cast(correct_prediction, 'float32'))

        ParameterServerModel.__init__(self, x, true_y, compute_gradients,
            apply_gradients, minimize, error_rate, session, batch_size)

    def process_data(self, data):
        batch_size = self.batch_size
        num_classes = self.get_num_classes()
        features = []
        labels = []
        label = [0] * num_classes
        if batch_size == 0:
            batch_size = 1000000
        for line in data:
            if len(line) is 0:
                print('Skipping empty line')
                continue
            split = line.split(',')
            split0 = int(split[0])
            if split0 >= num_classes:
                print('Error label out of range: %d' % split0)
                continue
            features.append([int(s) for s in split[1:]])
            label[split0] = 1
            labels.append(label)
            label[split0] = 0

        return labels, features

    """
    Doesn't do mini-batch, just loads (on each partition) `batch_size` number
    of data.
    """
    def process_partition(self, partition):
        batch_size = self.batch_size
        print('batch size = %d' % batch_size)
        num_classes = self.get_num_classes()
        features = []
        labels = []
        label = [0] * num_classes
        if batch_size == 0:
            batch_size = 1000000
        for i in range(batch_size):
            print('i = %d' % i)
            try:
                line = partition.next()
                if len(line) is 0:
                    print('Skipping empty line')
                    continue
                split = line.split(',')
                split0 = int(split[0])
                if split0 >= num_classes:
                    print('Error label out of range: %d' % split0)
                    continue
                features.append([int(s) for s in split[1:]])
                label[split0] = 1
                labels.append(label)
                label[split0] = 0
            except StopIteration:
                break

        return labels, features
