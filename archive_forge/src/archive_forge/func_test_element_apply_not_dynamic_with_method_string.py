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
def test_element_apply_not_dynamic_with_method_string(self):
    pinst = ParamClass()
    applied = self.element.apply('relabel', dynamic=False, label=pinst.param.label)
    self.assertEqual(applied, self.element.relabel('Test'))