import pickle
import warnings
from unittest import skipIf
import numpy as np
import pandas as pd
import param
import holoviews as hv
from holoviews.core.data import Dataset
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_astype_transform(self):
    expr = dim('int').astype('float64')
    self.assert_apply(expr, self.linear_ints.astype('float64'))