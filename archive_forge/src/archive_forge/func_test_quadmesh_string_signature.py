import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_quadmesh_string_signature(self):
    qmesh = QuadMesh(([0, 1], [0, 1], np.array([[0, 1], [0, 1]])), ['a', 'b'], 'c')
    self.assertEqual(qmesh.kdims, [Dimension('a'), Dimension('b')])
    self.assertEqual(qmesh.vdims, [Dimension('c')])