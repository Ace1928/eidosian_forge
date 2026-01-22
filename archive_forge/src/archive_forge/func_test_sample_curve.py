from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_sample_curve(self):
    curve_sampled = self.ds.to.curve('a', 'b', groupby=[]).sample([1, 2])
    curve_sampled2 = self.ds2.to.curve('a', 'b', groupby=[]).sample([1, 2])
    self.assertNotEqual(curve_sampled, curve_sampled2)
    self.assertEqual(curve_sampled.dataset, self.ds)
    ops = curve_sampled.pipeline.operations
    self.assertEqual(len(ops), 3)
    self.assertIs(ops[0].output_type, Dataset)
    self.assertIs(ops[1].output_type, Curve)
    self.assertEqual(ops[2].method_name, 'sample')
    self.assertEqual(ops[2].args, [[1, 2]])
    self.assertEqual(ops[2].kwargs, {})
    self.assertEqual(curve_sampled.pipeline(curve_sampled.dataset), curve_sampled)
    self.assertEqual(curve_sampled.pipeline(self.ds2), curve_sampled2)