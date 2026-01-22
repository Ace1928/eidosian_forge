from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_sort_curve(self):
    curve_sorted = self.ds.to.curve('a', 'b', groupby=[]).sort('a')
    curve_sorted2 = self.ds2.to.curve('a', 'b', groupby=[]).sort('a')
    self.assertNotEqual(curve_sorted, curve_sorted2)
    self.assertEqual(curve_sorted.dataset, self.ds)
    ops = curve_sorted.pipeline.operations
    self.assertEqual(len(ops), 3)
    self.assertIs(ops[0].output_type, Dataset)
    self.assertIs(ops[1].output_type, Curve)
    self.assertEqual(ops[2].method_name, 'sort')
    self.assertEqual(ops[2].args, ['a'])
    self.assertEqual(ops[2].kwargs, {})
    self.assertEqual(curve_sorted.pipeline(curve_sorted.dataset), curve_sorted)
    self.assertEqual(curve_sorted.pipeline(self.ds2), curve_sorted2)