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
def test_dmap_apply_method_as_string_with_instance_param(self):
    pinst = ParamClass()
    applied = self.dmap.apply('relabel', label=pinst.param.label)
    self.assertEqual(applied[1], self.dmap[1].relabel('Test'))
    pinst.label = 'Another label'
    self.assertEqual(applied[1], self.dmap[1].relabel('Another label'))