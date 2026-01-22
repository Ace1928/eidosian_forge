from unittest import mock
import keras_tuner
import numpy as np
import pytest
import tensorflow as tf
import autokeras as ak
from autokeras import test_utils
def dataset_error(x, y, validation_data, message, tmp_path):
    auto_model = get_multi_io_auto_model(tmp_path)
    with pytest.raises(ValueError) as info:
        auto_model.fit(x, y, epochs=2, validation_data=validation_data)
    assert message in str(info.value)