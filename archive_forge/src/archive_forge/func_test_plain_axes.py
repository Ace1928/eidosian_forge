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
def test_plain_axes(self):
    fig, ax = mpl.pyplot.subplots()
    fig.add_axes([0.2, 0.2, 0.2, 0.2])
    Series(np.random.default_rng(2).random(10)).plot(ax=ax)