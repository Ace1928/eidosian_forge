from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_hist_to_curve(self):
    curve = self.hist.to.curve()
    ops = curve.pipeline.operations
    self.assertEqual(len(ops), 4)
    self.assertIs(ops[0].output_type, Dataset)
    self.assertIs(ops[1].output_type, Apply)
    self.assertEqual(ops[2].method_name, '__call__')
    self.assertIsInstance(ops[2].args[0], histogram)
    self.assertIs(ops[3].output_type, Curve)
    self.assertEqual(curve.pipeline(curve.dataset), curve)