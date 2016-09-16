# parameterservermodel.py
import tensorflow as tf
import numpy as np
import time


class ParameterServerModel():
    """
    Args:
      x:                    features placeholder
      y:                    labels placeholder
      compute_gradients:    list of (gradient, variable)
      apply_gradients:      operation that applies gradients for a lst of (grad, var)
      minimize:             operation that computes & applies gradients
      error_rate:           tensor of prediction error
      session:              TensorFlow Session
      batch_size:           mini-batch size
    """
    def __init__(self, x, y, compute_gradients, apply_gradients, minimize,
          error_rate, session, batch_size):
        self.session = session
        self.batch_size = batch_size
        self.graph = session.graph
        self.session.graph.as_default().__enter__()
        self.x = x
        self.y = y
        self.compute_gradients = compute_gradients
        self.apply_gradients = apply_gradients
        self.error_rate = error_rate
        self.error_rate_summary = tf.scalar_summary('error_rate', error_rate)
        self.minimize = minimize
        self.reset_gradients()
        self.gradient_counter = tf.Variable(initial_value=0, trainable=False)

        self.parameter_assignments = [None] * len(self.compute_gradients)
        for i in range(len(self.compute_gradients)):
            gradient = self.compute_gradients[i][0]
            variable = self.compute_gradients[i][1]
            self.parameter_assignments[i] = variable.assign(gradient)
        self.session.run(tf.initialize_all_variables())

    def get_num_classes(self):
        return self.y.get_shape().as_list()[1]

    def train(self, labels, features):
        with self.session.as_default():
            #with self.graph.as_default():
            feed = { self.x: features, self.y: labels }
            for i in range(len(self.compute_gradients)):
                self.gradients[i] += self.compute_gradients[i][0].eval(
                    feed_dict=feed)

            self.num_gradients += 1
            #del feed

    def test(self, labels, features):
        with self.session.as_default():
            #with self.graph.as_default():
            feed = { self.x: features, self.y: labels }
            test_error_rate = self.error_rate.eval(feed_dict=feed)
            #del feed
            return test_error_rate

    def get_parameters(self):
        with self.session.as_default():
            #with self.graph.as_default():
            result = [None] * len(self.compute_gradients)
            for i in range(len(self.compute_gradients)):
                result[i] = self.compute_gradients[i][1].eval(
                    session=self.session)
            arr = np.array(result)
            #del result[:]
            #del result
            return arr

    def assign_parameters(self, parameters):
        with self.session.as_default():
            #with self.graph.as_default():
            self.reset_gradients()
            for i, grad_var in enumerate(self.compute_gradients):
                self.parameter_assignments[i].eval(
                    feed_dict={ grad_var[0]: parameters[i] })

    def apply(self, gradients):
        with self.graph.as_default():
            feed_dict = {}
            for i, grad_var in enumerate(self.compute_gradients):
                feed_dict[grad_var[0]] = gradients[i]

            self.apply_gradients.run(session=self.session, feed_dict=feed_dict)
            del feed_dict
            del gradients

    def get_gradients(self):
        result = [None] * (1 + len(self.gradients))
        for i in range(len(self.gradients)):
            result[i + 1] = np.divide(self.gradients[i], self.num_gradients) \
                .astype('float32')
        result[0] = [time.time()]
        arr = np.array(result)
        del result[:]
        del result
        return arr

    def reset_gradients(self):
        with self.session.as_default():
            self.gradients = [tf.zeros(g[1].get_shape()).eval()
                for g in self.compute_gradients]
            self.num_gradients = 0

    def train_warmup(self, partition, error_rates_filename):
        error_rates = []
        iteration = 0
        batch_size = self.batch_size
        for i in range(0, len(partition), batch_size):
            data = partition[i : i + batch_size]
            labels, features = self.process_data(data)
            if len(labels) is 0:
                break
            with self.session.as_default():
                feed = { self.x: features, self.y: labels }
                self.minimize.run(feed_dict=feed)
                error_rate = self.error_rate.eval(feed_dict=feed)
                t = time.time()
                with open(error_rates_filename, 'a') as f:
                    f.write('%f, %f\n' % (t, error_rate))
                error_rates.append(error_rate)
                iteration += 1
                print('Warmup training iteration %d at %f error_rate' % 
                    (iteration, error_rate))

        return error_rates

    def serialize(self, array):
        return array.dumps()

    def deserialize(self, serialized):
        return np.loads(serialized)
