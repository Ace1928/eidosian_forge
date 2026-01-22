import numpy
import pytest
from numpy.testing import assert_allclose
from thinc.api import (
from thinc.layers import chain, tuplify
def test_clone_noop():
    model = clone(Linear(), 0)
    assert len(model.layers) == 0
    assert model.name == 'noop'