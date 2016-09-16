# config.py
import mnistdnn
import os
import random

MASTER_IP = '172.31.1.230'
SPARK_MASTER_PORT = 7077
SPARK_APP_NAME = 'Herp Derp'
HDFS_PORT = 9000
HDFS_DIRECTORY = '/mnist/'
LOCAL_DIRECTORY = "/home/ubuntu/ssp-ml/"
ERROR_RATES_PATH = "/home/ubuntu/ssp-ml/errors.txt"
WEBSOCKET_PORT = 8123  # random.randint(30000, 60000)
MODEL_KEYWORD = 'mnist'

if MODEL_KEYWORD == 'mnist':
    TRAINING_RDD_FILENAME = ('hdfs://%s:%d' % (MASTER_IP, HDFS_PORT)) + \
        os.path.join(HDFS_DIRECTORY, 'mnist_train.csv')
    TEST_FILENAME = ('hdfs://%s:%d' % (MASTER_IP, HDFS_PORT)) + \
        os.path.join(HDFS_DIRECTORY, 'mnist_test.csv')
    LOCAL_TEST_PATH = os.path.join(LOCAL_DIRECTORY, 'mnist_test.csv')
    NUM_PARTITIONS = 3
    NUM_EPOCHS = 2
    WARMUP = 2000
    BATCH_SIZE = 50
    #BATCH_SIZE = 0
    EPOCHS = 5
    REPARTITION = True
    TIME_LAG = 100
    MODEL = mnistdnn.MnistDNN(BATCH_SIZE)
else:
    raise NotImplementedError('Currently only mnist model works')
