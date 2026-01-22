from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_clone_dataset_kwarg_none(self):
    ds_clone = self.ds.clone(dataset=None)
    self.assertIs(ds_clone, ds_clone.dataset)