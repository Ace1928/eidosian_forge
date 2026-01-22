import keras_tuner
import tensorflow as tf
from keras_tuner.engine import hyperparameters
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_int_seq_deserialize_to_int_seq():
    serialized_block = blocks.serialize(blocks.TextToIntSequence())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.TextToIntSequence)