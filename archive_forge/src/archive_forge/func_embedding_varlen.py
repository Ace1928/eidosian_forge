import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.layers.preprocessing.benchmarks import (
from tensorflow.python.eager.def_function import (
def embedding_varlen(batch_size, max_length):
    """Benchmark a variable-length embedding."""
    embedding_size = 32768
    data = fc_bm.create_data(max_length, batch_size * NUM_REPEATS, embedding_size - 1, dtype=int)
    model = keras.Sequential()
    model.add(keras.Input(shape=(None,), ragged=True, name='data', dtype=tf.int64))
    model.add(keras.layers.Embedding(embedding_size, 256))
    model.add(keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1)))
    fc = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_identity('data', num_buckets=embedding_size - 1), dimension=256)

    @tf_function()
    def fc_fn(tensors):
        fc.transform_feature(tf.__internal__.feature_column.FeatureTransformationCache(tensors), None)
    keras_data = {'data': data}
    k_avg_time = fc_bm.run_keras(keras_data, model, batch_size, NUM_REPEATS)
    fc_data = {'data': data.to_sparse()}
    fc_avg_time = fc_bm.run_fc(fc_data, fc_fn, batch_size, NUM_REPEATS)
    return (k_avg_time, fc_avg_time)