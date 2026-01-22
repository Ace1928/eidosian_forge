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
def test_donot_overwrite_index_name(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((2, 2)), columns=['a', 'b'])
    df.index.name = 'NAME'
    df.plot(y='b', label='LABEL')
    assert df.index.name == 'NAME'