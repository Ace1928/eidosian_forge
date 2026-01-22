from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_bad_label(self):
    df = DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    msg = 'label should be list-like and same length as y'
    with pytest.raises(ValueError, match=msg):
        df.plot(x='A', y=['B', 'C'], label='bad_label')