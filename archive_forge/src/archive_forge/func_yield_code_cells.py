from __future__ import annotations
import hashlib
import os
import sys
import typing as t
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
from hmac import HMAC
from pathlib import Path
from base64 import encodebytes
from jupyter_core.application import JupyterApp, base_flags
from traitlets import Any, Bool, Bytes, Callable, Enum, Instance, Integer, Unicode, default, observe
from traitlets.config import LoggingConfigurable, MultipleInstanceError
from . import NO_CONVERT, __version__, read, reads
def yield_code_cells(nb):
    """Iterator that yields all cells in a notebook

    nbformat version independent
    """
    if nb.nbformat >= 4:
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                yield cell
    elif nb.nbformat == 3:
        for ws in nb['worksheets']:
            for cell in ws['cells']:
                if cell['cell_type'] == 'code':
                    yield cell