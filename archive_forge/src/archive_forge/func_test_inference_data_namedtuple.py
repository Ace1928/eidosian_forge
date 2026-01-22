from collections import namedtuple
import numpy as np
import pytest
from ...data.io_numpyro import from_numpyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
def test_inference_data_namedtuple(self, data):
    samples = data.obj.get_samples()
    Samples = namedtuple('Samples', samples)
    data_namedtuple = Samples(**samples)
    _old_fn = data.obj.get_samples
    data.obj.get_samples = lambda *args, **kwargs: data_namedtuple
    inference_data = from_numpyro(posterior=data.obj)
    assert isinstance(data.obj.get_samples(), Samples)
    data.obj.get_samples = _old_fn
    for key in samples:
        assert key in inference_data.posterior