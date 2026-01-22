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
@property
def plotlyjs(self):
    """
        The plotly.js bundle being used for image rendering.

        Returns
        -------
        str
        """
    return self._constants.get('plotlyjs', None)