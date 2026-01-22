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
def raise_format_value_error(val):
    raise ValueError('\nInvalid value of type {typ} receive as an image format specification.\n    Received value: {v}\n\nAn image format must be specified as one of the following string values:\n    {valid_formats}'.format(typ=type(val), v=val, valid_formats=sorted(format_conversions.keys())))