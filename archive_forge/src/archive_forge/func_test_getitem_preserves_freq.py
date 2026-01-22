import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.indexes.common import Base
def test_getitem_preserves_freq(self, simple_index):
    index = simple_index
    assert index.freq is not None
    result = index[:]
    assert result.freq == index.freq