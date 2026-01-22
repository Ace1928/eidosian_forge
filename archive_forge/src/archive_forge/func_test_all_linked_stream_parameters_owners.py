from collections import defaultdict
from unittest import SkipTest
import pandas as pd
import param
import pytest
from panel.widgets import IntSlider
import holoviews as hv
from holoviews.core.spaces import DynamicMap
from holoviews.core.util import Version
from holoviews.element import Curve, Histogram, Points, Polygons, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import *  # noqa (Test all available streams)
from holoviews.util import Dynamic, extension
from holoviews.util.transform import dim
from .utils import LoggingComparisonTestCase
def test_all_linked_stream_parameters_owners():
    """Test to ensure operations can accept parameters in streams dictionary"""
    stream_classes = param.concrete_descendents(LinkedStream)
    for stream_class in stream_classes.values():
        for name, p in stream_class.param.objects().items():
            if name != 'name' and p.owner != stream_class:
                msg = 'Linked stream %r has parameter %r which is inherited from %s. Parameter needs to be redeclared in the class definition of this linked stream.'
                raise Exception(msg % (stream_class, name, p.owner))