import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def monotonic_index(start, end, dtype='int64', closed='right'):
    return IntervalIndex.from_breaks(np.arange(start, end, dtype=dtype), closed=closed)