from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def test_ConditionalNeqSys3():
    _check_NaCl(_get_cneqsys3(-60), [None], 4, method='lm')