import numpy as np
import pytest
import tensorflow as tf
from keras_tuner.engine import hyperparameters
from autokeras.utils import utils
def test_get_hyperparameter_with_hp_return_same():
    hp = utils.get_hyperparameter(hyperparameters.Choice('hp', [10, 30]), hyperparameters.Choice('hp', [10, 20]), int)
    assert isinstance(hp, hyperparameters.Choice)