import pytest
from bokeh.plotting import figure
from panel.layout import Row
from panel.links import Link
from panel.pane import Bokeh, HoloViews
from panel.tests.util import hv_available
from panel.widgets import (
@hv_available
def test_link_with_customcode(document, comm):
    range_widget = RangeSlider(start=0.0, end=1.0)
    curve = hv.Curve([])
    code = '\n      x_range.start = source.value[0]\n      x_range.end = source.value[1]\n    '
    range_widget.jslink(curve, code={'value': code})
    row = Row(HoloViews(curve, backend='bokeh'), range_widget)
    range_widget.value = (0.5, 0.7)
    model = row.get_root(document, comm=comm)
    hv_views = row.select(HoloViews)
    widg_views = row.select(RangeSlider)
    assert len(hv_views) == 1
    assert len(widg_views) == 1
    range_slider = widg_views[0]._models[model.ref['id']][0]
    x_range = hv_views[0]._plots[model.ref['id']][0].handles['x_range']
    link_customjs = range_slider.js_property_callbacks['change:value'][-1]
    assert link_customjs.args['source'] is range_slider
    assert link_customjs.args['x_range'] is x_range
    assert link_customjs.code == 'try { %s } catch(err) { console.log(err) }' % code