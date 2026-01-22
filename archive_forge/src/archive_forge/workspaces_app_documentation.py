from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
from typing import Any
from jupyter_core.application import JupyterApp
from traitlets import Bool, Unicode
from ._version import __version__
from .config import LabConfig
from .workspaces_handler import WorkspacesManager
Start the app.