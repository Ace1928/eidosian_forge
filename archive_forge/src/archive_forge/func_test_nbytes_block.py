import re
import numpy as np
import pytest
from pandas._libs.sparse import IntIndex
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
def test_nbytes_block(self):
    arr = SparseArray([1, 2, 0, 0, 0], kind='block')
    result = arr.nbytes
    assert result == 24