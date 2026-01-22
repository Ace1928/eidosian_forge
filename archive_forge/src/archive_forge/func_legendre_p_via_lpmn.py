import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
def legendre_p_via_lpmn(n, x):
    return lpmn(0, n, x)[0][0, -1]