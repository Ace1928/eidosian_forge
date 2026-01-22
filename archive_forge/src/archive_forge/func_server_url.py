import atexit
import json
import os
import socket
import subprocess
import sys
import threading
import warnings
from copy import copy
from contextlib import contextmanager
from pathlib import Path
from shutil import which
import tenacity
import plotly
from plotly.files import PLOTLY_DIR, ensure_writable_plotly_dir
from plotly.io._utils import validate_coerce_fig_to_dict
from plotly.optional_imports import get_module
@server_url.setter
def server_url(self, val):
    if val is None:
        self._props.pop('server_url', None)
        return
    if not isinstance(val, str):
        raise ValueError('\nThe server_url property must be a string, but received value of type {typ}.\n    Received value: {val}'.format(typ=type(val), val=val))
    if not val.startswith('http://') and (not val.startswith('https://')):
        val = 'http://' + val
    shutdown_server()
    self.executable = None
    self.port = None
    self.timeout = None
    self.mathjax = None
    self.topojson = None
    self.mapbox_access_token = None
    self._props['server_url'] = val