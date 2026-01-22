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
@pytest.mark.parametrize('kind', plotting.PlotAccessor._common_kinds)
def test_all_invalid_plot_data(self, kind):
    df = DataFrame(list('abcd'))
    msg = 'no numeric data to plot'
    with pytest.raises(TypeError, match=msg):
        df.plot(kind=kind)