from collections import (
from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('mapping', [list, defaultdict, []])
def test_to_dict_errors(self, mapping):
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 3)))
    msg = '|'.join(["unsupported type: <class 'list'>", 'to_dict\\(\\) only accepts initialized defaultdicts'])
    with pytest.raises(TypeError, match=msg):
        df.to_dict(into=mapping)