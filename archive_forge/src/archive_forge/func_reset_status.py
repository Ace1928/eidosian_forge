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
def reset_status():
    """
    Shutdown the running orca server, if any, and reset the orca status
    to unvalidated.

    This command is only needed if the desired orca executable is changed
    during an interactive session.

    Returns
    -------
    None
    """
    shutdown_server()
    status._props['executable_list'] = None
    status._props['version'] = None
    status._props['state'] = 'unvalidated'