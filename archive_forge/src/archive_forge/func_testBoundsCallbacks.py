import uuid
from unittest import TestCase
from unittest.mock import Mock
import plotly.graph_objs as go
from holoviews import Tiles
from holoviews.plotting.plotly.callbacks import (
from holoviews.streams import (
def testBoundsCallbacks(self):
    bounds_classes = [BoundsXYCallback, BoundsXCallback, BoundsYCallback]
    xyplots, xystreamss, xycallbacks, xyevents = build_callback_set(BoundsXYCallback, ['first', 'second', 'third', 'forth', 'other'], BoundsXY, 2)
    xplots, xstreamss, xcallbacks, xevents = build_callback_set(BoundsXCallback, ['first', 'second', 'third', 'forth', 'other'], BoundsX, 2)
    yplots, ystreamss, ycallbacks, yevents = build_callback_set(BoundsYCallback, ['first', 'second', 'third', 'forth', 'other'], BoundsY, 2)
    selected_data1 = {'range': {'x': [1, 4], 'y': [-1, 5]}}
    for cb_cls in bounds_classes:
        cb_cls.update_streams_from_property_update('selected_data', selected_data1, self.fig_dict)
    for xystream, xstream, ystream in zip(xystreamss[0] + xystreamss[1], xstreamss[0] + xstreamss[1], ystreamss[0] + ystreamss[1]):
        assert xystream.bounds == (1, -1, 4, 5)
        assert xstream.boundsx == (1, 4)
        assert ystream.boundsy == (-1, 5)
    for xystream, xstream, ystream in zip(xystreamss[2] + xystreamss[3], xstreamss[2] + xstreamss[3], ystreamss[2] + ystreamss[3]):
        assert xystream.bounds is None
        assert xstream.boundsx is None
        assert ystream.boundsy is None
    selected_data2 = {'range': {'x2': [2, 5], 'y2': [0, 6]}}
    for cb_cls in bounds_classes:
        cb_cls.update_streams_from_property_update('selected_data', selected_data2, self.fig_dict)
    for xystream, xstream, ystream in zip(xystreamss[2], xstreamss[2], ystreamss[2]):
        assert xystream.bounds == (2, 0, 5, 6)
        assert xstream.boundsx == (2, 5)
        assert ystream.boundsy == (0, 6)
    selected_data3 = {'range': {'x3': [3, 6], 'y3': [1, 7]}}
    for cb_cls in bounds_classes:
        cb_cls.update_streams_from_property_update('selected_data', selected_data3, self.fig_dict)
    for xystream, xstream, ystream in zip(xystreamss[3], xstreamss[3], ystreamss[3]):
        assert xystream.bounds == (3, 1, 6, 7)
        assert xstream.boundsx == (3, 6)
        assert ystream.boundsy == (1, 7)
    selected_data_lasso = {'lassoPoints': {'x': [1, 4, 2], 'y': [-1, 5, 2]}}
    for cb_cls in bounds_classes:
        cb_cls.update_streams_from_property_update('selected_data', selected_data_lasso, self.fig_dict)
    for xystream, xstream, ystream in zip(xystreamss[0] + xystreamss[1] + xystreamss[2] + xystreamss[3], xstreamss[0] + xstreamss[1] + xstreamss[2] + xstreamss[3], ystreamss[0] + ystreamss[1] + ystreamss[2] + ystreamss[3]):
        assert xystream.bounds is None
        assert xstream.boundsx is None
        assert ystream.boundsy is None
    for xyevent, xevent, yevent in zip(xyevents[4], xevents[4], yevents[4]):
        assert xyevent == []
        assert xevent == []
        assert yevent == []