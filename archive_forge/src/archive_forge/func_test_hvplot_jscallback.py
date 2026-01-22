import pytest
from bokeh.plotting import figure
from panel.layout import Row
from panel.links import Link
from panel.pane import Bokeh, HoloViews
from panel.tests.util import hv_available
from panel.widgets import (
@hv_available
def test_hvplot_jscallback(document, comm):
    points1 = hv.Points([1, 2, 3])
    hvplot = HoloViews(points1, backend='bokeh')
    hvplot.jscallback(**{'x_range.start': 'some_code'})
    model = hvplot.get_root(document, comm=comm)
    x_range = hvplot._plots[model.ref['id']][0].handles['x_range']
    customjs = x_range.js_property_callbacks['change:start'][-1]
    assert customjs.args['source'] is x_range
    assert customjs.code == 'try { some_code } catch(err) { console.log(err) }'