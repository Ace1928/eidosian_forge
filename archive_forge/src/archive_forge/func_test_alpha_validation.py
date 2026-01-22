from datetime import datetime
import io
import itertools
import re
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
from matplotlib.collections import (Collection, LineCollection,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_alpha_validation(pcfunc):
    fig, ax = plt.subplots()
    pc = getattr(ax, pcfunc)(np.arange(12).reshape((3, 4)))
    with pytest.raises(ValueError, match='^Data array shape'):
        pc.set_alpha([0.5, 0.6])
        pc.update_scalarmappable()