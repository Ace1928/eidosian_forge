from statsmodels.compat.python import lrange
from io import BytesIO
from itertools import product
import numpy as np
from numpy.testing import assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.api import datasets
from statsmodels.graphics.mosaicplot import (
@pytest.mark.matplotlib
def test_default_arg_index(close_figures):
    df = pd.DataFrame({'size': ['small', 'large', 'large', 'small', 'large', 'small'], 'length': ['long', 'short', 'short', 'long', 'long', 'short']})
    assert_raises(ValueError, mosaic, data=df, title='foobar')