import keras_tuner
import tensorflow as tf
from keras_tuner.engine import hyperparameters
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_int_seq_build_with_seq_len_return_tensor():
    block = blocks.TextToIntSequence(output_sequence_length=50)
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(1,), dtype=tf.string))
    assert len(nest.flatten(outputs)) == 1