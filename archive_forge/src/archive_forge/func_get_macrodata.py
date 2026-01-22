from statsmodels.compat.pandas import QUARTER_END, assert_index_equal
from statsmodels.compat.python import lrange
from io import BytesIO, StringIO
import os
import sys
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
import statsmodels.tools.data as data_util
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VAR, var_acf
def get_macrodata():
    data = macrodata.load_pandas().data[['realgdp', 'realcons', 'realinv']]
    data = data.to_records(index=False)
    nd = data.view((float, 3), type=np.ndarray)
    nd = np.diff(np.log(nd), axis=0)
    return nd.ravel().view(data.dtype, type=np.ndarray)