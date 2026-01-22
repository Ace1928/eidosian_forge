import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_temporal_global_avg_return_tensor():
    block = blocks.TemporalReduction(reduction_type='global_avg')
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 10), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1