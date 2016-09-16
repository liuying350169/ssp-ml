# tensorspark.py
import config
import parameterwebsocketclient
from operator import add
import tornado.web
import tornado.websocket
import tornado.ioloop
import threading
import tensorflow as tf
import pyspark


class ParameterServerWebsocketHandler(tornado.websocket.WebSocketHandler):
    def __init__(self, *args, **kwargs):
        self.server = kwargs.pop('server')
        self.model = self.server.model
        with self.model.session.graph.as_default():
            self.saver = tf.train.Saver()
        self.lock = threading.Lock()
        super(ParameterServerWebsocketHandler, self).__init__(*args, **kwargs)

    def open(self):
        self.send_parameters()

    def on_close(self):
        pass

    def on_message(self, message):
        # Message is the serialized gradient & time
        time_gradient = self.model.deserialize(message)
        self.server.gradient_count += 1
        #print('gradient count: %d' % self.server.gradient_count)
        time_sent = time_gradient[0][0]

        # Update (with atomic lock) and only if the gradient came "on-time"
        if time.time() - time_sent < TIME_LAG:
            self.lock.acquire()
            gradient = time_gradient[1:]
            self.model.apply(gradient)
            # Monitoring
            if self.server.gradient_count % 10 == 0:
                error_rate = self.model.test(self.server.test_labels,
                    self.server.test_features)
                print('gradients received: %d    error_rate: %f' %
                    (self.server.gradient_count, error_rate))
                with open(config.ERROR_RATES_PATH, 'a') as f:
                    f.write('%f, %d, %f\n' %
                        (t, self.server.gradient_count, error_rate))
            self.lock.release()
        else:
            print("rejected")
        del time_gradient
        self.send_parameters()

    def send_parameters(self):
        # Send the (updated?) parameters to the client
        self.lock.acquire()
        parameters = self.model.get_parameters()
        self.lock.release()
        serialized = self.model.serialize(parameters)
        self.write_message(serialized, binary=True)

class ParameterServer(threading.Thread):
    def __init__(self, model, warmup_data=None, test_data=None):
        threading.Thread.__init__(self)
        self.model = model
        self.test_labels, self.test_features = \
            self.model.process_data(test_data)
        self.warmup(warmup_data)
        self.gradient_count = 0
        self.application = tornado.web.Application(
            [(r'/', ParameterServerWebsocketHandler, {'server': self})])

    def warmup(self, data=None):
        if data is not None:
            self.model.train_warmup(partition=data,
                error_rates_filename=config.ERROR_RATES_PATH)

    def run(self):
        self.application.listen(config.WEBSOCKET_PORT)
        tornado.ioloop.IOLoop.current().start()

def train_partition(partition):
    return parameterserverwebsocketclient.TensorSparkWorker(
        config.MODEL_KEYWORD,
        config.BATCH_SIZE, config.WEBSOCKET_PORT).train_partition(partition)

def test_partition(partition):
    return parameterserverwebsocketclient.TensorSparkWorker(
        config.MODEL_KEYWORD,
        config.BATCH_SIZE, config.WEBSOCKET_PORT).test_partition(partition)

def train_epochs(num_epochs, training_rdd, num_partitions):
    for i in range(num_epochs):
        print('training epoch %d' % i)
        if REPARTITION:
            training_rdd = training_rdd.repartition(num_partitions)
        mapped_training = training_rdd.mapPartitions(train_partition)
        # mapped_training should return `num_partitions` []'s
        mapped_training.take(10)

def test_all_partitions(sc):
    testing_rdd = sc.textFile(config.TEST_FILENAME).cache()
    mapped_testing = testing_rdd.mapPartitions(test_partition)
    # mapped_testing should return `num_partitions` [error_rate]'s
    return mapped_testing.reduce(add)


conf = pyspark.SparkConf() \
    .setMaster("spark://%s:%d" %
        (config.SPARK_MASTER_IP, config.SPARK_MASTER_PORT))
    .setAppName(config.SPARK_APP_NAME)
sc = pyspark.SparkContext()
try:
    training_rdd = sc.textFile(config.TRAINING_RDD_FILENAME)
    print('num partitions = %s' % training_rdd.getNumPartitions())

    warmup_data = training_rdd.take(config.WARMUP)

    with open(config.LOCAL_TEST_PATH, 'r') as test_file:
        test_data_lines = test_file.readlines()

    with open(config.ERROR_RATES_PATH, 'w') as f:
        f.write('')

    test_data = test_data_lines[0 : 100]

    parameter_server = ts.ParameterServer(config.MODEL,
        warmup_data,
        test_data)

    ts.train_epochs(config.NUM_EPOCHS, training_rdd, config.NUM_PARTITIONS)

finally:
    tornado.ioloop.IOLoop.current().stop()
