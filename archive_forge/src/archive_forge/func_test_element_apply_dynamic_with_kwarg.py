import numpy as np
import pandas as pd
import param
import pytest
from panel.widgets import IntSlider, RadioButtonGroup, TextInput
from holoviews import Dataset, util
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import Curve, Image
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import ParamMethod, Params
def test_element_apply_dynamic_with_kwarg(self):
    applied = self.element.apply(lambda x, label: x.relabel(label), dynamic=True, label='Test')
    self.assertEqual(len(applied.streams), 0)
    self.assertEqual(applied[()], self.element.relabel('Test'))