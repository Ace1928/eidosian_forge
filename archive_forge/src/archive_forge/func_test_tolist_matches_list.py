import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas._testing as tm
def test_tolist_matches_list(self, index):
    assert index.tolist() == list(index)