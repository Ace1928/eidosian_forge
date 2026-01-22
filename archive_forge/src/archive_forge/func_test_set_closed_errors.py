from itertools import permutations
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
@pytest.mark.parametrize('bad_closed', ['foo', 10, 'LEFT', True, False])
def test_set_closed_errors(self, bad_closed):
    index = interval_range(0, 5)
    msg = f"invalid option for 'closed': {bad_closed}"
    with pytest.raises(ValueError, match=msg):
        index.set_closed(bad_closed)