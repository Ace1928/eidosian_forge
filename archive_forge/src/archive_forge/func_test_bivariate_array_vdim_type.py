import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_bivariate_array_vdim_type(self):
    dist = Bivariate(np.array([[0, 1], [1, 2], [2, 3]]))
    assert dist.get_dimension_type(2) == np.float64