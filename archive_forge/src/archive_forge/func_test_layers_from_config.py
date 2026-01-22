from typing import List, Optional
import numpy
import pytest
import srsly
from numpy.testing import assert_almost_equal
from thinc.api import Dropout, Model, NumpyOps, registry, with_padded
from thinc.backends import NumpyOps
from thinc.compat import has_torch
from thinc.types import Array2d, Floats2d, FloatsXd, Padded, Ragged, Shape
from thinc.util import data_validation, get_width
@pytest.mark.parametrize('name,kwargs,in_data,out_data', TEST_CASES)
def test_layers_from_config(name, kwargs, in_data, out_data):
    cfg = {'@layers': name, **kwargs}
    filled_cfg = registry.fill({'config': cfg})
    assert srsly.is_json_serializable(filled_cfg)
    model = registry.resolve({'config': cfg})['config']
    if 'LSTM' in name:
        model = with_padded(model)
    valid = True
    with data_validation(valid):
        model.initialize(in_data, out_data)
        Y, backprop = model(in_data, is_train=True)
        if model.has_dim('nO'):
            assert get_width(Y) == model.get_dim('nO')
        assert_data_match(Y, out_data)
        dX = backprop(Y)
        assert_data_match(dX, in_data)
        model._to_ops(NoDropoutOps())
        model.predict(in_data)