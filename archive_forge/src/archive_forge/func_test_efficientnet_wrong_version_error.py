import keras_tuner
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils
def test_efficientnet_wrong_version_error():
    with pytest.raises(ValueError) as info:
        blocks.EfficientNetBlock(version='abc')
    assert 'Expect version to be' in str(info.value)