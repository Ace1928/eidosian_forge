from unittest import TestCase, SkipTest
import sys
from parameterized import parameterized
import numpy as np
import pandas as pd
from holoviews.core import GridMatrix, NdOverlay
from holoviews.element import (
from hvplot import scatter_matrix
@parameterized.expand([('spread',), ('dynspread',)])
def test_spread_kwargs(self, operation):
    sm = scatter_matrix(self.df, datashade=True, **{operation: True, 'shape': 'circle'})
    dm = sm['a', 'b']
    dm[()]
    self.assertEqual(dm.last.pipeline.operations[-1].args[0].keywords['shape'], 'circle')