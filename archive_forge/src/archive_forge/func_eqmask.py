from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def eqmask(m1, m2):
    if m1 is nomask:
        return m2 is nomask
    if m2 is nomask:
        return m1 is nomask
    return (m1 == m2).all()