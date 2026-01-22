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
def test_null_collection_datalim():
    col = mcollections.PathCollection([])
    col_data_lim = col.get_datalim(mtransforms.IdentityTransform())
    assert_array_equal(col_data_lim.get_points(), mtransforms.Bbox.null().get_points())