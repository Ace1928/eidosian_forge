import os
from pathlib import Path
import bokeh
from packaging.version import Version
from panel.config import config, panel_extension as extension
from panel.io.resources import (
from panel.io.state import set_curdoc
from panel.theme.native import Native
from panel.widgets import Button
def test_resources_model_inline(document):
    resources = Resources(mode='inline')
    with set_resource_mode('inline'):
        with set_curdoc(document):
            extension('tabulator')
            assert resources.js_raw[-2:] == [(DIST_DIR / 'bundled/datatabulator/tabulator-tables@5.5.0/dist/js/tabulator.min.js').read_text(encoding='utf-8'), (DIST_DIR / 'bundled/datatabulator/luxon/build/global/luxon.min.js').read_text(encoding='utf-8')]
            assert resources.css_raw == [(DIST_DIR / 'bundled/datatabulator/tabulator-tables@5.5.0/dist/css/tabulator_simple.min.css').read_text(encoding='utf-8')]