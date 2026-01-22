import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
def gdtrix_comp(b, p):
    return gdtrix(1.0, b, 1 - p)