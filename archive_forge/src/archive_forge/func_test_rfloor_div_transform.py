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
def test_rfloor_div_transform(self):
    expr = 2 // dim('int')
    self.assert_apply(expr, 2 // self.linear_ints)