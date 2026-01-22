from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multiindex_get_loc_list_raises(self):
    idx = MultiIndex.from_tuples([('a', 1), ('b', 2)])
    msg = '\\[\\]'
    with pytest.raises(InvalidIndexError, match=msg):
        idx.get_loc([])