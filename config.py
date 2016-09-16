# config.py
import mnistdnn
import os
import random

HDFS_DIRECTORY = 'PATH_TO_HDFS_DIR'
LOCAL_DIRECTORY = "PATH_TO_LOCAL_DIR"
ERROR_RATES_PATH = "PATH_TO_ERROR_RATES"
WEBSOCKET_PORT = random.randint(30000, 60000)  # or 30303
MODEL_KEYWORD = 'mnist'

if MODEL_KEYWORD == 'mnist':
    TRAINING_RDD_FILENAME = os.path.join(HDFS_DIRECTORY, 'mnist_train.csv')
    TEST_FILENAME = os.path.join(HDFS_DIRECTORY, 'mnist_test.csv')
    LOCAL_TEST_PATH = os.path.join(LOCAL_DIRECTORY, 'mnist_test.csv')
    PARTITIONS = 48
    WARMUP = 2000
    BATCH_SIZE = 50
    EPOCHS = 5
    REPARTITION = True
    TIME_LAG = 100
    MODEL = mnistdnn.MnistDNN(BATCH_SIZE)
else:
    raise NotImplementedError('Currently only mnist model works')

NUM_EPOCHS = 2
NUM_PARTITIONS = 3