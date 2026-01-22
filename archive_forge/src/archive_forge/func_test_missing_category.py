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
def test_missing_category(close_figures):
    animal = ['dog', 'dog', 'dog', 'cat', 'dog', 'cat', 'cat', 'dog', 'dog', 'cat']
    size = ['medium', 'large', 'medium', 'medium', 'medium', 'medium', 'large', 'large', 'large', 'small']
    testdata = pd.DataFrame({'animal': animal, 'size': size})
    testdata['size'] = pd.Categorical(testdata['size'], categories=['small', 'medium', 'large'])
    testdata = testdata.sort_values('size')
    fig, _ = mosaic(testdata, ['animal', 'size'])
    bio = BytesIO()
    fig.savefig(bio, format='png')