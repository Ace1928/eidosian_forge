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
def test_multi_dim_expression(self):
    expr = dim('int') - dim('float')
    self.assert_apply(expr, self.linear_ints - self.linear_floats)