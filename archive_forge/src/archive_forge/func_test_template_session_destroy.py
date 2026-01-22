import json
import param
import pytest
from bokeh.document import Document
from bokeh.io.doc import patch_curdoc
from panel.layout import GridSpec, Row
from panel.pane import HoloViews, Markdown
from panel.template import (
from panel.template.base import BasicTemplate
from panel.widgets import FloatSlider
from .util import hv_available
def test_template_session_destroy(document, comm):
    tmplt = Template(template)
    widget = FloatSlider()
    row = Row('A', 'B')
    tmplt.add_panel('A', widget)
    tmplt.add_panel('B', row)
    tmplt._init_doc(document, comm, notebook=True)
    session_context = param.Parameterized()
    session_context._document = document
    session_context.id = 'Some ID'
    assert len(widget._models) == 2
    assert len(row._models) == 2
    assert len(row[0]._models) == 2
    assert len(row[1]._models) == 2
    for cb in document.session_destroyed_callbacks:
        cb(session_context)
    assert len(widget._models) == 0
    assert len(row._models) == 0
    assert len(row[0]._models) == 0
    assert len(row[1]._models) == 0