from unittest import TestCase, SkipTest
import sys
from parameterized import parameterized
import numpy as np
import pandas as pd
from holoviews.core import GridMatrix, NdOverlay
from holoviews.element import (
from hvplot import scatter_matrix
@parameterized.expand([('rasterize',), ('datashade',)])
def test_datashade_aggregator(self, operation):
    sm = scatter_matrix(self.df, aggregator='mean', **{operation: True})
    dm = sm['a', 'b']
    dm[()]
    self.assertEqual(dm.last.pipeline.operations[-1].aggregator, 'mean')