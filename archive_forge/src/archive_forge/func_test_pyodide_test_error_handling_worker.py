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
def test_pyodide_test_error_handling_worker(http_serve, page):
    wait_for_app(http_serve, error_app, page, 'pyodide-worker', wait=False)
    expect(page.locator('.pn-loading-msg')).to_have_text('RuntimeError: This app is broken', timeout=TIMEOUT)