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
def test_collection_set_verts_array():
    verts = np.arange(80, dtype=np.double).reshape(10, 4, 2)
    col_arr = PolyCollection(verts)
    col_list = PolyCollection(list(verts))
    assert len(col_arr._paths) == len(col_list._paths)
    for ap, lp in zip(col_arr._paths, col_list._paths):
        assert np.array_equal(ap._vertices, lp._vertices)
        assert np.array_equal(ap._codes, lp._codes)
    verts_tuple = np.empty(10, dtype=object)
    verts_tuple[:] = [tuple((tuple(y) for y in x)) for x in verts]
    col_arr_tuple = PolyCollection(verts_tuple)
    assert len(col_arr._paths) == len(col_arr_tuple._paths)
    for ap, atp in zip(col_arr._paths, col_arr_tuple._paths):
        assert np.array_equal(ap._vertices, atp._vertices)
        assert np.array_equal(ap._codes, atp._codes)