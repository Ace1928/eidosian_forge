import logging
import numpy as np
import pandas as pd
from param import get_logger
from holoviews.core.data import Dataset, MultiInterface
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
def test_varying_values_and_scalar_not_isscalar_per_geom(self):
    path = Path([{'x': [1, 2, 3, 4, 5], 'y': [0, 0, 1, 1, 2], 'value': np.arange(5)}, {'x': [5, 4, 3, 2, 1], 'y': [2, 2, 1, 1, 0], 'value': 1}], vdims='value', datatype=[self.datatype])
    self.assertIs(path.interface, self.interface)
    self.assertFalse(path.interface.isscalar(path, 'value', per_geom=True))