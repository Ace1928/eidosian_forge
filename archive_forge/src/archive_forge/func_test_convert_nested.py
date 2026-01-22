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
def test_convert_nested(self, pc, axis):
    data = ['2012-1-1', '2012-1-2']
    r1 = pc.convert([data, data], None, axis)
    r2 = [pc.convert(data, None, axis) for _ in range(2)]
    assert r1 == r2