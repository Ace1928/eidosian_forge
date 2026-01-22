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
def test_mosaic_very_complex(close_figures):
    key_name = ['gender', 'age', 'health', 'work']
    key_base = (['male', 'female'], ['old', 'young'], ['healty', 'ill'], ['work', 'unemployed'])
    keys = list(product(*key_base))
    data = dict(zip(keys, range(1, 1 + len(keys))))
    props = {}
    props['male', 'old'] = {'color': 'r'}
    props['female',] = {'color': 'pink'}
    L = len(key_base)
    _, axes = plt.subplots(L, L)
    for i in range(L):
        for j in range(L):
            m = set(range(L)).difference({i, j})
            if i == j:
                axes[i, i].text(0.5, 0.5, key_name[i], ha='center', va='center')
                axes[i, i].set_xticks([])
                axes[i, i].set_xticklabels([])
                axes[i, i].set_yticks([])
                axes[i, i].set_yticklabels([])
            else:
                ji = max(i, j)
                ij = min(i, j)
                temp_data = {(k[ij], k[ji]) + tuple((k[r] for r in m)): v for k, v in data.items()}
                keys = list(temp_data.keys())
                for k in keys:
                    value = _reduce_dict(temp_data, k[:2])
                    temp_data[k[:2]] = value
                    del temp_data[k]
                mosaic(temp_data, ax=axes[i, j], axes_label=False, properties=props, gap=0.05, horizontal=i > j)
    plt.suptitle('old males should look bright red,  (plot 4 of 4)')