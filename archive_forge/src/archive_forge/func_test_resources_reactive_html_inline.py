import os
from pathlib import Path
import bokeh
from packaging.version import Version
from panel.config import config, panel_extension as extension
from panel.io.resources import (
from panel.io.state import set_curdoc
from panel.theme.native import Native
from panel.widgets import Button
def test_resources_reactive_html_inline(document):
    resources = Resources(mode='inline')
    with set_resource_mode('inline'):
        with set_curdoc(document):
            extension('gridstack')
            assert resources.js_raw[-1:] == [(DIST_DIR / 'bundled/gridstack/gridstack@7.2.3/dist/gridstack-all.js').read_text(encoding='utf-8')]
            assert resources.css_raw == [(DIST_DIR / 'bundled/gridstack/gridstack@7.2.3/dist/gridstack.min.css').read_text(encoding='utf-8'), (DIST_DIR / 'bundled/gridstack/gridstack@7.2.3/dist/gridstack-extra.min.css').read_text(encoding='utf-8')]