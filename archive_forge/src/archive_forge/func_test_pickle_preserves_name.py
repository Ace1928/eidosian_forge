import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas._testing as tm
def test_pickle_preserves_name(self, index):
    original_name, index.name = (index.name, 'foo')
    unpickled = tm.round_trip_pickle(index)
    assert index.equals(unpickled)
    index.name = original_name