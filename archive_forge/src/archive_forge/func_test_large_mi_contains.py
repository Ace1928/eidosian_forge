from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_large_mi_contains(self, monkeypatch):
    with monkeypatch.context():
        monkeypatch.setattr(libindex, '_SIZE_CUTOFF', 10)
        result = MultiIndex.from_arrays([range(10), range(10)])
        assert (10, 0) not in result