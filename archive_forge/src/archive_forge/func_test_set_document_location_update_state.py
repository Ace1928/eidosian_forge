import time
import pytest
import panel as pn
from panel.tests.util import serve_component, wait_until
from panel.util import parse_query
from panel.widgets import FloatSlider, RangeSlider, TextInput
def test_set_document_location_update_state(page):
    widget = TextInput(name='Text')

    def app():
        if pn.state.location:
            pn.state.location.sync(widget, {'value': 'text_value'})

        def cb():
            """Do nothing callback"""
            assert pn.state.location.search == '?text_value=Text+Value'
        pn.state.onload(cb)
        return widget
    serve_component(page, app, suffix='/?text_value=Text+Value')
    wait_until(lambda: widget.value == 'Text Value', page)