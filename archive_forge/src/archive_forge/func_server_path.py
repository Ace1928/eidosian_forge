from __future__ import annotations
import logging # isort:skip
from pathlib import Path
from .deprecation import deprecated
def server_path() -> Path:
    """ Get the location of the server subpackage.

    """
    return ROOT_DIR / 'server'