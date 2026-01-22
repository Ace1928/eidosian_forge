from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_reindex_curve(self):
    curve_ba = self.ds.to(Curve, 'a', 'b', groupby=[]).reindex(kdims='b', vdims='a')
    curve2_ba = self.ds2.to(Curve, 'a', 'b', groupby=[]).reindex(kdims='b', vdims='a')
    self.assertNotEqual(curve_ba, curve2_ba)
    self.assertEqual(curve_ba.dataset, self.ds)
    ops = curve_ba.pipeline.operations
    self.assertEqual(len(ops), 3)
    self.assertIs(ops[0].output_type, Dataset)
    self.assertIs(ops[1].output_type, Curve)
    self.assertEqual(ops[2].method_name, 'reindex')
    self.assertEqual(ops[2].args, [])
    self.assertEqual(ops[2].kwargs, dict(kdims='b', vdims='a'))
    self.assertEqual(curve_ba.pipeline(curve_ba.dataset), curve_ba)
    self.assertEqual(curve_ba.pipeline(self.ds2), curve2_ba)