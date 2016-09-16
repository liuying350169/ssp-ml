# main.py
import config
import pyspark
import tensorspark as ts

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
