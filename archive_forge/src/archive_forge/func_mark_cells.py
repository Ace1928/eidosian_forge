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
def mark_cells(self, nb, trusted):
    """Mark cells as trusted if the notebook's signature can be verified

        Sets ``cell.metadata.trusted = True | False`` on all code cells,
        depending on the *trusted* parameter. This will typically be the return
        value from ``self.check_signature(nb)``.

        This function is the inverse of check_cells
        """
    if nb.nbformat < 3:
        return
    for cell in yield_code_cells(nb):
        cell['metadata']['trusted'] = trusted