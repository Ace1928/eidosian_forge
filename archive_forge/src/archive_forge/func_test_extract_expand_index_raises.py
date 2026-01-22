from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
def test_extract_expand_index_raises():
    idx = Index(['A1', 'A2', 'A3', 'A4', 'B5'])
    msg = 'only one regex group is supported with Index'
    with pytest.raises(ValueError, match=msg):
        idx.str.extract('([AB])([123])', expand=False)