import numpy as np
import packaging
import pytest
from ...data.io_pyro import from_pyro  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
def test_inference_data_only_posterior(self, data):
    idata = from_pyro(data.obj)
    test_dict = {'posterior': ['mu', 'tau', 'eta'], 'sample_stats': ['diverging']}
    fails = check_multiple_attrs(test_dict, idata)
    assert not fails