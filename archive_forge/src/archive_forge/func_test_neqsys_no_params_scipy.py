from __future__ import (absolute_import, division, print_function)
import math
import pytest
import numpy as np
from .. import NeqSys, ConditionalNeqSys, ChainedNeqSys
def test_neqsys_no_params_scipy():
    _test_neqsys_no_params('scipy')