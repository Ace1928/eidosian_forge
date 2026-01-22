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
def test_multi_operator_expression_repr(self):
    self.assertEqual(repr(((dim('float') - 2) * 3) ** 2), "((dim('float')-2)*3)**2")