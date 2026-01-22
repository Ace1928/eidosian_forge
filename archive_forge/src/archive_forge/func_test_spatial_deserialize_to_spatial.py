import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_spatial_deserialize_to_spatial():
    serialized_block = blocks.serialize(blocks.SpatialReduction())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.SpatialReduction)