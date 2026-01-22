import numpy
import pytest
from hypothesis import given
from thinc.api import Padded, Ragged, get_width
from thinc.types import ArgsKwargs
from thinc.util import (
from . import strategies
@pytest.mark.parametrize('obj', [1234, 'foo', {'a': numpy.array(0)}])
def test_get_width_fail(obj):
    with pytest.raises(ValueError):
        get_width(obj)