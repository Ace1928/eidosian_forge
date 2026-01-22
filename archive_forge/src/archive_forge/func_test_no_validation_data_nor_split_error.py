from unittest import mock
import keras_tuner
import numpy as np
import pytest
import tensorflow as tf
import autokeras as ak
from autokeras import test_utils
def test_no_validation_data_nor_split_error(tmp_path):
    auto_model = ak.AutoModel(ak.ImageInput(), ak.RegressionHead(), directory=tmp_path)
    with pytest.raises(ValueError) as info:
        auto_model.fit(x=np.random.rand(100, 32, 32, 3), y=np.random.rand(100, 1), validation_split=0)
    assert 'Either validation_data or a non-zero' in str(info.value)