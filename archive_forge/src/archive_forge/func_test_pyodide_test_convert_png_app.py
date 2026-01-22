import os
import pathlib
import re
import shutil
import sys
import tempfile
import time
import uuid
from subprocess import PIPE, Popen
import pytest
from playwright.sync_api import expect
from panel.config import config
from panel.io.convert import BOKEH_LOCAL_WHL, PANEL_LOCAL_WHL, convert_apps
import panel as pn
import panel as pn
import panel as pn
import panel as pn
import pandas as pd
import pandas as pd
import sys
import panel as pn
import panel as pn
import panel as pn
import panel as pn
@pytest.mark.parametrize('runtime', ['pyodide', 'pyodide-worker'])
def test_pyodide_test_convert_png_app(http_serve, page, runtime):
    msgs = wait_for_app(http_serve, png_app, page, runtime)
    expect(page.locator('img')).to_have_count(1)
    assert [msg for msg in msgs if msg.type == 'error' and 'favicon' not in msg.location['url']] == []