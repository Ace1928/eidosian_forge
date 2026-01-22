import time
import pytest
import panel as pn
from panel.tests.util import serve_component, wait_until
from panel.util import parse_query
from panel.widgets import FloatSlider, RangeSlider, TextInput
def test_set_hash_update_document(page):

    def app():
        """simple app to set hash at onload"""
        widget = TextInput(name='Text')

        def cb():
            pn.state.location.hash = '#123'
        pn.state.onload(cb)
        return widget
    _, port = serve_component(page, app)
    expected_location = {'href': f'http://localhost:{port}/#123', 'protocol': 'http:', 'hostname': 'localhost', 'port': f'{port}', 'pathname': '/', 'search': '', 'hash': '#123', 'reload': None}
    verify_document_location(expected_location, page)