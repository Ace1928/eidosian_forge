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
def test_line_label_none(self):
    s = Series([1, 2])
    ax = s.plot()
    assert ax.get_legend() is None
    ax = s.plot(legend=True)
    assert ax.get_legend().get_texts()[0].get_text() == ''