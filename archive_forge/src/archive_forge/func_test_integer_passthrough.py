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
def test_integer_passthrough(self, pc, axis):
    rs = pc.convert([0, 1], None, axis)
    xp = [0, 1]
    assert rs == xp