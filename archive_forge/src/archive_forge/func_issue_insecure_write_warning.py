from __future__ import annotations
import errno
import os
import site
import stat
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional
import platformdirs
from .utils import deprecation
def issue_insecure_write_warning() -> None:
    """Issue an insecure write warning."""

    def format_warning(msg: str, *args: Any, **kwargs: Any) -> str:
        return str(msg) + '\n'
    warnings.formatwarning = format_warning
    warnings.warn("WARNING: Insecure writes have been enabled via environment variable 'JUPYTER_ALLOW_INSECURE_WRITES'! If this is not intended, remove the variable or set its value to 'False'.", stacklevel=2)