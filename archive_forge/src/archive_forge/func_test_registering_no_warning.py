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
def test_registering_no_warning(self):
    plt = pytest.importorskip('matplotlib.pyplot')
    s = Series(range(12), index=date_range('2017', periods=12))
    _, ax = plt.subplots()
    register_matplotlib_converters()
    ax.plot(s.index, s.values)
    plt.close()