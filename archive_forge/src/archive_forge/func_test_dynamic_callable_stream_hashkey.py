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
def test_dynamic_callable_stream_hashkey(self):
    history = deque(maxlen=10)

    def history_callback(x):
        if x is not None:
            history.append(x)
        return Curve(list(history))

    class NoMemoize(PointerX):
        x = param.ClassSelector(class_=pointer_types, default=None, constant=True)

        @property
        def hashkey(self):
            return {'hash': uuid.uuid4().hex}
    x = NoMemoize()
    dmap = DynamicMap(history_callback, kdims=[], streams=[x])
    x.add_subscriber(lambda **kwargs: dmap[()])
    x.event(x=1)
    x.event(x=1)
    self.assertEqual(dmap[()], Curve([1, 1, 1]))
    x.event(x=2)
    x.event(x=2)
    self.assertEqual(dmap[()], Curve([1, 1, 1, 2, 2, 2]))