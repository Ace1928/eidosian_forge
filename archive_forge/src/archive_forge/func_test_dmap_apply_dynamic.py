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
def test_dmap_apply_dynamic(self):
    applied = self.dmap.apply(lambda x: x.relabel('Test'))
    self.assertEqual(len(applied.streams), 0)
    self.assertEqual(applied[1], self.dmap[1].relabel('Test'))