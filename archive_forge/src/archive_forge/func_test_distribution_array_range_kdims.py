import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_distribution_array_range_kdims(self):
    dist = Distribution(np.array([0, 1, 2]))
    assert dist.range(0) == (0, 2)