import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dropna_corner(self, float_frame):
    msg = 'invalid how option: foo'
    with pytest.raises(ValueError, match=msg):
        float_frame.dropna(how='foo')
    with pytest.raises(KeyError, match="^\\['X'\\]$"):
        float_frame.dropna(subset=['A', 'X'])