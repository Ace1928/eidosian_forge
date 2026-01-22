import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('fdata', fdata, ids=fids)
def test_non_string_fails(self, fdata):
    with pytest.raises(TypeError):
        cat.UnitData(fdata)