from copy import deepcopy
import inspect
import pydoc
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._config.config import option_context
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_pop_assign_name(self, float_frame):
    s = float_frame['A']
    assert s.name == 'A'
    s = float_frame.pop('A')
    assert s.name == 'A'
    s = float_frame.loc[:, 'B']
    assert s.name == 'B'
    s2 = s.loc[:]
    assert s2.name == 'B'