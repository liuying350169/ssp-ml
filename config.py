# config.py
import mnistdnn
import os
import random

MASTER_IP = '172.31.1.230'
SPARK_MASTER_PORT = 7077
SPARK_APP_NAME = 'Herp Derp'
HDFS_PORT = 9000
HDFS_DIRECTORY = '/mnist/'
LOCAL_DIRECTORY = "/home/ubuntu/"
ERROR_RATES_PATH = "/home/ubuntu/errors.txt"
WEBSOCKET_PORT = random.randint(30000, 60000)  # or 30303
MODEL_KEYWORD = 'mnist'

if MODEL_KEYWORD == 'mnist':
    TRAINING_RDD_FILENAME = os.path.join('hdfs://%s:%d' % (MASTER_IP, HDFS_PORT),
        HDFS_DIRECTORY, 'mnist_train.csv')
    TEST_FILENAME = os.path.join('hdfs://%s:%d' % (MASTER_IP, HDFS_PORT),
        HDFS_DIRECTORY, 'mnist_test.csv')
    LOCAL_TEST_PATH = os.path.join(LOCAL_DIRECTORY, 'mnist_test.csv')
    NUM_PARTITIONS = 48
    NUM_EPOCHS = 2
    WARMUP = 2000
    BATCH_SIZE = 50
    EPOCHS = 5
    REPARTITION = True
    TIME_LAG = 100
    MODEL = mnistdnn.MnistDNN(BATCH_SIZE)
else:
    raise NotImplementedError('Currently only mnist model works')
