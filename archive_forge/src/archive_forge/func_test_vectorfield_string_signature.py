import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_vectorfield_string_signature(self):
    vectorfield = VectorField([], ['a', 'b'], ['c', 'd'])
    self.assertEqual(vectorfield.kdims, [Dimension('a'), Dimension('b')])
    self.assertEqual(vectorfield.vdims, [Dimension('c'), Dimension('d')])