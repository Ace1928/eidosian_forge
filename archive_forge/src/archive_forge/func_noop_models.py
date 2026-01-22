import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
@pytest.fixture
def noop_models():
    return [with_padded(noop()), with_array(noop()), with_array2d(noop()), with_list(noop()), with_ragged(noop())]