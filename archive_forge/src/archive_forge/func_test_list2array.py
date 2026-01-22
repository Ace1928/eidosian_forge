import numpy
import pytest
from thinc.api import NumpyOps, Ragged, registry, strings2arrays
from ..util import get_data_checker
def test_list2array(list_data, array_data):
    check_transform('list2array.v1', list_data, array_data)