import os
from pathlib import Path
import bokeh
from packaging.version import Version
from panel.config import config, panel_extension as extension
from panel.io.resources import (
from panel.io.state import set_curdoc
from panel.theme.native import Native
from panel.widgets import Button
def test_resources_config_css_files(document):
    resources = Resources(mode='cdn')
    with set_curdoc(document):
        config.css_files = [Path(__file__).parent.parent / 'assets' / 'custom.css']
        assert resources.css_raw == ['/* Test */\n']