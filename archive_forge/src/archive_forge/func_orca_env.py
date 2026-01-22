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
@contextmanager
def orca_env():
    """
    Context manager to clear and restore environment variables that are
    problematic for orca to function properly

    NODE_OPTIONS: When this variable is set, orca <v1.2 will have a
    segmentation fault due to an electron bug.
    See: https://github.com/electron/electron/issues/12695

    ELECTRON_RUN_AS_NODE: When this environment variable is set the call
    to orca is transformed into a call to nodejs.
    See https://github.com/plotly/orca/issues/149#issuecomment-443506732
    """
    clear_env_vars = ['NODE_OPTIONS', 'ELECTRON_RUN_AS_NODE', 'LD_PRELOAD']
    orig_env_vars = {}
    try:
        orig_env_vars.update({var: os.environ.pop(var) for var in clear_env_vars if var in os.environ})
        yield
    finally:
        for var, val in orig_env_vars.items():
            os.environ[var] = val