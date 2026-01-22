import os
from pathlib import Path
import bokeh
from packaging.version import Version
from panel.config import config, panel_extension as extension
from panel.io.resources import (
from panel.io.state import set_curdoc
from panel.theme.native import Native
from panel.widgets import Button
def test_resources_design_server(document):
    resources = Resources(mode='server')
    with set_resource_mode('server'):
        with set_curdoc(document):
            extension(design='bootstrap')
            assert resources.js_files[-1:] == ['static/extensions/panel/bundled/bootstrap5/js/bootstrap.bundle.min.js']