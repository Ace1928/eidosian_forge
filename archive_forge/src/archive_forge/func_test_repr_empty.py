from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_repr_empty(self):
    repr(DataFrame())
    frame = DataFrame(index=np.arange(1000))
    repr(frame)