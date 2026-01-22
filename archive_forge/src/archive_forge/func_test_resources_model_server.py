import os
from pathlib import Path
import bokeh
from packaging.version import Version
from panel.config import config, panel_extension as extension
from panel.io.resources import (
from panel.io.state import set_curdoc
from panel.theme.native import Native
from panel.widgets import Button
def test_resources_model_server(document):
    resources = Resources(mode='server')
    with set_resource_mode('server'):
        with set_curdoc(document):
            extension('tabulator')
            assert resources.js_files[:2] == ['static/extensions/panel/bundled/datatabulator/tabulator-tables@5.5.0/dist/js/tabulator.min.js', 'static/extensions/panel/bundled/datatabulator/luxon/build/global/luxon.min.js']
            assert resources.css_files == [f'static/extensions/panel/bundled/datatabulator/tabulator-tables@5.5.0/dist/css/tabulator_simple.min.css?v={JS_VERSION}']