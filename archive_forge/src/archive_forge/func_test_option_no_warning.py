from datetime import (
import subprocess
import sys
import numpy as np
import pytest
import pandas._config.config as cf
from pandas._libs.tslibs import to_offset
from pandas import (
import pandas._testing as tm
from pandas.plotting import (
from pandas.tseries.offsets import (
def test_option_no_warning(self):
    pytest.importorskip('matplotlib.pyplot')
    ctx = cf.option_context('plotting.matplotlib.register_converters', False)
    plt = pytest.importorskip('matplotlib.pyplot')
    s = Series(range(12), index=date_range('2017', periods=12))
    _, ax = plt.subplots()
    with ctx:
        ax.plot(s.index, s.values)
    register_matplotlib_converters()
    with ctx:
        ax.plot(s.index, s.values)
    plt.close()