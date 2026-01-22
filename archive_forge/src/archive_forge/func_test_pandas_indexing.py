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
def test_pandas_indexing(pd):
    index = [11, 12, 13]
    ec = fc = pd.Series(['red', 'blue', 'green'], index=index)
    lw = pd.Series([1, 2, 3], index=index)
    ls = pd.Series(['solid', 'dashed', 'dashdot'], index=index)
    aa = pd.Series([True, False, True], index=index)
    Collection(edgecolors=ec)
    Collection(facecolors=fc)
    Collection(linewidths=lw)
    Collection(linestyles=ls)
    Collection(antialiaseds=aa)