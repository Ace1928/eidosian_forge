import os
from pathlib import Path
import bokeh
from packaging.version import Version
from panel.config import config, panel_extension as extension
from panel.io.resources import (
from panel.io.state import set_curdoc
from panel.theme.native import Native
from panel.widgets import Button
def test_resources_server_absolute():
    resources = Resources(mode='server', absolute=True, minified=True)
    assert resources.js_raw == ['Bokeh.set_log_level("info");']
    assert resources.js_files == ['http://localhost:5006/static/js/bokeh.min.js', 'http://localhost:5006/static/js/bokeh-gl.min.js', 'http://localhost:5006/static/js/bokeh-widgets.min.js', 'http://localhost:5006/static/js/bokeh-tables.min.js', 'http://localhost:5006/static/js/bokeh-mathjax.min.js']