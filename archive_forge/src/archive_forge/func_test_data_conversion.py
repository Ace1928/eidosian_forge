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
def test_data_conversion(close_figures):
    import pandas
    _, ax = plt.subplots(4, 4)
    data = {'ax': 1, 'bx': 2, 'cx': 3}
    mosaic(data, ax=ax[0, 0], title='basic dict', axes_label=False)
    data = pandas.Series(data)
    mosaic(data, ax=ax[0, 1], title='basic series', axes_label=False)
    data = [1, 2, 3]
    mosaic(data, ax=ax[0, 2], title='basic list', axes_label=False)
    data = np.asarray(data)
    mosaic(data, ax=ax[0, 3], title='basic array', axes_label=False)
    plt.close('all')
    data = {('ax', 'cx'): 1, ('bx', 'cx'): 2, ('ax', 'dx'): 3, ('bx', 'dx'): 4}
    mosaic(data, ax=ax[1, 0], title='compound dict', axes_label=False)
    mosaic(data, ax=ax[2, 0], title='inverted keys dict', index=[1, 0], axes_label=False)
    data = pandas.Series(data)
    mosaic(data, ax=ax[1, 1], title='compound series', axes_label=False)
    mosaic(data, ax=ax[2, 1], title='inverted keys series', index=[1, 0])
    data = [[1, 2], [3, 4]]
    mosaic(data, ax=ax[1, 2], title='compound list', axes_label=False)
    mosaic(data, ax=ax[2, 2], title='inverted keys list', index=[1, 0])
    data = np.array([[1, 2], [3, 4]])
    mosaic(data, ax=ax[1, 3], title='compound array', axes_label=False)
    mosaic(data, ax=ax[2, 3], title='inverted keys array', index=[1, 0], axes_label=False)
    plt.close('all')
    gender = ['male', 'male', 'male', 'female', 'female', 'female']
    pet = ['cat', 'dog', 'dog', 'cat', 'dog', 'cat']
    data = pandas.DataFrame({'gender': gender, 'pet': pet})
    mosaic(data, ['gender'], ax=ax[3, 0], title='dataframe by key 1', axes_label=False)
    mosaic(data, ['pet'], ax=ax[3, 1], title='dataframe by key 2', axes_label=False)
    mosaic(data, ['gender', 'pet'], ax=ax[3, 2], title='both keys', axes_label=False)
    mosaic(data, ['pet', 'gender'], ax=ax[3, 3], title='keys inverted', axes_label=False)
    plt.close('all')
    plt.suptitle('testing data conversion (plot 1 of 4)')