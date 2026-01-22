from __future__ import (absolute_import, division, print_function)
from functools import reduce
import inspect
import math
import operator
import sys
from pkg_resources import parse_requirements, parse_version
import numpy as np
import pytest
def pycvodes_double(cb):
    try:
        from pycvodes import config
        prec = config.get('SUNDIALS_PRECISION', 'double')
    except:
        prec = 'double'
    r = 'Test is designed only for pycvodes built with double precision.'
    return pytest.mark.skipif(prec != 'double', reason=r)(cb)