import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.genmod.qif import (QIF, QIFIndependence, QIFExchangeable,
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.genmod import families
def llf_gn(params):
    return model.objective(params)[3]