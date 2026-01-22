import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import analysers
from autokeras import blocks
from autokeras import test_utils
def test_image_deserialize_to_image():
    serialized_block = blocks.serialize(blocks.ImageBlock())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.ImageBlock)