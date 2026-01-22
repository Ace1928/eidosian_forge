import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_bivariate_array_constructor_custom_vdim(self):
    dist = Bivariate(np.array([[0, 1, 2], [0, 1, 2]]), vdims=['Test'])
    assert dist.kdims == [Dimension('x'), Dimension('y')]
    assert dist.vdims == [Dimension('Test')]