import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def iter_multi_index(i):
    ret = []
    while not i.finished:
        ret.append(i.multi_index)
        i.iternext()
    return ret