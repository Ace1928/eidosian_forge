import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
def test_tuplify_one(model1):
    with pytest.raises(TypeError):
        tuplify(model1)