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
def test_plain_axes_make_inset_axes(self):
    fig, ax = mpl.pyplot.subplots()
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    iax = inset_axes(ax, width='30%', height=1.0, loc=3)
    Series(np.random.default_rng(2).random(10)).plot(ax=ax)
    Series(np.random.default_rng(2).random(10)).plot(ax=iax)