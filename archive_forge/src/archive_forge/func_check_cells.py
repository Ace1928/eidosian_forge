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
def check_cells(self, nb):
    """Return whether all code cells are trusted.

        A cell is trusted if the 'trusted' field in its metadata is truthy, or
        if it has no potentially unsafe outputs.
        If there are no code cells, return True.

        This function is the inverse of mark_cells.
        """
    if nb.nbformat < 3:
        return False
    trusted = True
    for cell in yield_code_cells(nb):
        if not self._check_cell(cell, nb.nbformat):
            trusted = False
    return trusted