from datetime import timedelta
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
import pandas._testing as tm
def test_getitem_ambiguous_keyerror(indexer_sl):
    ser = Series(range(10), index=list(range(0, 20, 2)))
    with pytest.raises(KeyError, match='^1$'):
        indexer_sl(ser)[1]