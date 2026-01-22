import numpy as np
import pandas as pd
import pytest
from holoviews.core.dimension import Dimension
from holoviews.core.options import Compositor, Store
from holoviews.element import (
def test_bivariate_composite_filled(self):
    dist = Bivariate(np.random.rand(10, 2)).opts(filled=True)
    contours = Compositor.collapse_element(dist, backend='matplotlib')
    assert isinstance(contours, Polygons)
    assert contours.vdims[0].name == 'Density'