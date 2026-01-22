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
def test_mosaic_simple(close_figures):
    key_set = (['male', 'female'], ['old', 'adult', 'young'], ['worker', 'unemployed'], ['healty', 'ill'])
    keys = list(product(*key_set))
    data = dict(zip(keys, range(1, 1 + len(keys))))
    props = {}
    props['male',] = {'color': 'b'}
    props['female',] = {'color': 'r'}
    for key in keys:
        if 'ill' in key:
            if 'male' in key:
                props[key] = {'color': 'BlueViolet', 'hatch': '+'}
            else:
                props[key] = {'color': 'Crimson', 'hatch': '+'}
    mosaic(data, gap=0.05, properties=props, axes_label=False)
    plt.suptitle('syntetic data, 4 categories (plot 2 of 4)')