import numpy as np
import pytest
import tensorflow as tf
from keras_tuner.engine import hyperparameters
from autokeras.utils import utils
def test_get_hyperparameter_with_int_return_int():
    value = utils.get_hyperparameter(10, hyperparameters.Choice('hp', [10, 20]), int)
    assert isinstance(value, int)
    assert value == 10