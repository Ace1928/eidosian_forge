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
def test_reverse_binary_op_repr(self):
    self.assertEqual(repr(1 + dim('float')), "1+dim('float')")