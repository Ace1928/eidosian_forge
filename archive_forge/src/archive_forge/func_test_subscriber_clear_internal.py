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
def test_subscriber_clear_internal(self):
    pointerx = PointerX(x=2)
    pointerx.add_subscriber(self.fn1, precedence=0)
    pointerx.add_subscriber(self.fn2, precedence=1)
    pointerx.add_subscriber(self.fn3, precedence=1.5)
    pointerx.add_subscriber(self.fn4, precedence=10)
    self.assertEqual(pointerx.subscribers, [self.fn1, self.fn2, self.fn3, self.fn4])
    pointerx.clear('internal')
    self.assertEqual(pointerx.subscribers, [self.fn1, self.fn2])