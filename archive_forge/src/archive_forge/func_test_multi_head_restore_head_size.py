import keras_tuner
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_multi_head_restore_head_size():
    block = blocks.basic.MultiHeadSelfAttention(head_size=16)
    block = blocks.basic.MultiHeadSelfAttention.from_config(block.get_config())
    assert block.head_size == 16