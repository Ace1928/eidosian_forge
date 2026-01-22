import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
def gdtrib_comp(p, x):
    return gdtrib(1.0, 1 - p, x)