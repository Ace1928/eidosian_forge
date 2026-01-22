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
@image_comparison(['EventCollection_plot__switch_orientation__2x'])
def test__EventCollection__switch_orientation_2x():
    """
    Check that calling switch_orientation twice sets the orientation back to
    the default.
    """
    splt, coll, props = generate_EventCollection_plot()
    coll.switch_orientation()
    coll.switch_orientation()
    new_positions = coll.get_positions()
    assert props['orientation'] == coll.get_orientation()
    assert coll.is_horizontal()
    np.testing.assert_array_equal(props['positions'], new_positions)
    check_segments(coll, new_positions, props['linelength'], props['lineoffset'], props['orientation'])
    splt.set_title('EventCollection: switch_orientation 2x')