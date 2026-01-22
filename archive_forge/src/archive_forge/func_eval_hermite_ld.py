import os
import numpy as np
from numpy.testing import suppress_warnings
import pytest
from scipy.special import (
from scipy.integrate import IntegrationWarning
from scipy.special._testutils import FuncData
def eval_hermite_ld(n, x):
    return eval_hermite(n.astype('l'), x)