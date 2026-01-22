import numpy
import numpy.testing
import pytest
from thinc.api import (
from thinc.types import Padded, Ragged
from ..util import get_data_checker
@pytest.fixture
def padded_input(ops, list_input):
    return ops.list2padded(list_input)