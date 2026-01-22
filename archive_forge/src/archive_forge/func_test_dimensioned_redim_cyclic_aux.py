import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_dimensioned_redim_cyclic_aux(self):
    dimensioned = Dimensioned('Arbitrary Data', kdims=['x'])
    redimensioned = dimensioned.redim.cyclic(x=True)
    self.assertEqual(redimensioned.kdims[0].cyclic, True)