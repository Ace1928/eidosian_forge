from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def test_x_by_name():
    powell_sys = NeqSys(2, f=_powell, names=['u', 'v'], x_by_name=True)
    _test_powell(zip([powell_sys] * 2, [None, 'mpmath']), x0={'u': 1, 'v': 1})