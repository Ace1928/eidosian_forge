import keras_tuner
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_transformer_get_config_has_all_attributes():
    block = blocks.Transformer()
    config = block.get_config()
    assert test_utils.get_func_args(blocks.Transformer.__init__).issubset(config.keys())