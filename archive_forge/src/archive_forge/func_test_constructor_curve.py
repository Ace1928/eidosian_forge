from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_constructor_curve(self):
    element = Curve(self.df)
    expected = Dataset(self.df, kdims=self.df.columns[0], vdims=self.df.columns[1:].tolist())
    self.assertEqual(element.dataset, expected)
    pipeline = element.pipeline
    self.assertEqual(len(pipeline.operations), 1)
    self.assertIs(pipeline.operations[0].output_type, Curve)
    self.assertEqual(element, element.pipeline(element.dataset))