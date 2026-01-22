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
@pytest.mark.single_cpu
def test_registry_mpl_resets():
    code = 'import matplotlib.units as units; import matplotlib.dates as mdates; n_conv = len(units.registry); import pandas as pd; pd.plotting.register_matplotlib_converters(); pd.plotting.deregister_matplotlib_converters(); assert len(units.registry) == n_conv'
    call = [sys.executable, '-c', code]
    subprocess.check_output(call)