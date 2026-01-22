import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
def cexpm1(x, y):
    z = expm1(x + 1j * y)
    return (z.real, z.imag)