from datetime import datetime
from itertools import chain
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
def test_errorbar_plot_invalid_yerr_shape(self):
    s = Series(np.arange(10), name='x')
    with tm.external_error_raised(ValueError):
        s.plot(yerr=np.arange(11))