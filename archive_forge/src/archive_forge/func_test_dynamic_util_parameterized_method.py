import time
import uuid
from collections import deque
import numpy as np
import param
import pytest
from holoviews import Dimension, GridSpace, Layout, NdLayout, NdOverlay
from holoviews.core.options import Store
from holoviews.core.spaces import Callable, DynamicMap, HoloMap
from holoviews.element import Curve, Image, Points, Scatter, Text
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import histogram
from holoviews.plotting.util import initialize_dynamic
from holoviews.streams import (
from holoviews.util import Dynamic
from ..utils import LoggingComparisonTestCase
from .test_dimensioned import CustomBackendTestCase, ExampleElement
def test_dynamic_util_parameterized_method(self):

    class Test(param.Parameterized):
        label = param.String(default='test')

        @param.depends('label')
        def apply_label(self, obj):
            return obj.relabel(self.label)
    test = Test()
    dmap = Dynamic(self.dmap, operation=test.apply_label)
    test.label = 'custom label'
    self.assertEqual(dmap[0, 3].label, 'custom label')