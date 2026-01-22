import copy
import matplotlib.pyplot as plt
from matplotlib.scale import (
import matplotlib.scale as mscale
from matplotlib.ticker import AsinhLocator, LogFormatterSciNotation
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import numpy as np
from numpy.testing import assert_allclose
import io
import pytest
def test_scale_deepcopy():
    sc = mscale.LogScale(axis='x', base=10)
    sc2 = copy.deepcopy(sc)
    assert str(sc.get_transform()) == str(sc2.get_transform())
    assert sc._transform is not sc2._transform