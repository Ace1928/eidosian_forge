import os
from pathlib import Path
import bokeh
from packaging.version import Version
from panel.config import config, panel_extension as extension
from panel.io.resources import (
from panel.io.state import set_curdoc
from panel.theme.native import Native
from panel.widgets import Button
def test_resources_reactive_html_cdn(document):
    resources = Resources(mode='cdn')
    with set_resource_mode('cdn'):
        with set_curdoc(document):
            extension('gridstack')
            assert resources.js_files[-1:] == [f'{CDN_DIST}bundled/gridstack/gridstack@7.2.3/dist/gridstack-all.js']
            assert resources.css_files == [f'{CDN_DIST}bundled/gridstack/gridstack@7.2.3/dist/gridstack.min.css?v={JS_VERSION}', f'{CDN_DIST}bundled/gridstack/gridstack@7.2.3/dist/gridstack-extra.min.css?v={JS_VERSION}']