# parameterserverwebsocketclient.py
import config
import tornado.websocket
from tornado import gen
from tornado.ioloop import IOLoop
import tensorflow as tf

class Borg:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state


class TensorSparkWorker(Borg):
    def __init__(self, batch_size, websocket_port):
        Borg.__init__(self)

        if 'model' not in self.__dict__:
            print('Creating new Borg worker')
            self.model = config.MODEL
        self.batch_size = batch_size
        self.websocket_port = websocket_port
        self.loop = IOLoop.current()
        self.loop.run_sync(self.init_websocket)
        self.iteration = 0

    @gen.coroutine
    def init_websocket(self):
        self.websock = yield tornado.websocket.websocket_connect(
            "ws://%s:%d/" % (config.MASTER_IP, self.websocket_port),
            #"ws://52.26.64.207:50505/",
            connect_timeout=5)

    def train_partition(self, partition):
        with open('/home/ubuntu/blah.txt', 'w') as f: f.write('hi')
        while True:
            #with open('/home/ubuntu/blah.txt', 'a') as f: f.write('hello')
            labels, features = self.model.process_partition(partition)

            if len(labels) is 0:
                break

            if self.time_to_pull(self.iteration):
                self.request_parameters()
            print('starting training')
            self.model.train(labels, features)
            print('done training small batch')
            self.iteration += 1

            if self.time_to_push(self.iteration):
                self.push_gradients()
            print('blahg')

        # "null" value (Spark Master doesn't care about what it returns)
        return []

    def test_partition(self, partition):
        labels, features = self.model.process_partition(partition)
        self.request_parameters()
        error_rate = self.model.test(labels, features)
        return [error_rate]

    def test(self, data):
        if len(data) is 0:
            return 1.0
        self.request_parameters()
        accuracy = self.model.test(data)
        return accuracy

    def request_parameters(self):
        IOLoop.current().run_sync(self.request_parameters_coroutine)

    @gen.coroutine
    def request_parameters_coroutine(self):
        parameters = yield self.websock.read_message()
        parameters = self.model.deserialize(parameters)

    def time_to_pull(self, iteration):
        return iteration % 5 == 0

    def time_to_push(self, iteration):
        return iteration % 5 == 0

    def push_gradients(self):
        IOLoop.current().run_sync(self.push_gradients_coroutine)

    @gen.coroutine
    def push_gradients_coroutine(self):
        serialized = self.model.serialize(self.model.get_gradients())
        self.websock.write_message(serialized, binary=True)

