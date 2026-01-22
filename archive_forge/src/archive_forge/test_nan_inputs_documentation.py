from typing import Callable
import numpy as np
from numpy.testing import assert_array_equal, assert_, suppress_warnings
import pytest
import scipy.special as sc
Test how the ufuncs in special handle nan inputs.

