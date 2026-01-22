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
def test_convert_accepts_unicode(self, pc, axis):
    r1 = pc.convert('2012-1-1', None, axis)
    r2 = pc.convert('2012-1-1', None, axis)
    assert r1 == r2