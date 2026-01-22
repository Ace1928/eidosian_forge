import pytest
from pandas import DataFrame
from pandas.tests.plotting.common import (
def test_colorbar_layout(self):
    fig = plt.figure()
    axes = fig.subplot_mosaic('\n            AB\n            CC\n            ')
    x = [1, 2, 3]
    y = [1, 2, 3]
    cs0 = axes['A'].scatter(x, y)
    axes['B'].scatter(x, y)
    fig.colorbar(cs0, ax=[axes['A'], axes['B']], location='right')
    DataFrame(x).plot(ax=axes['C'])