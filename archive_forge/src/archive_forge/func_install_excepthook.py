from __future__ import annotations
import functools
import logging
import os
import subprocess
import sys
import warnings
from datetime import datetime
from typing import TYPE_CHECKING
def install_excepthook(hook_type: str='color', **kwargs) -> int:
    """
    This function replaces the original python traceback with an improved
    version from Ipython. Use `color` for colourful traceback formatting,
    `verbose` for Ka-Ping Yee's "cgitb.py" version kwargs are the keyword
    arguments passed to the constructor. See IPython.core.ultratb.py for more
    info.

    Returns:
        0 if hook is installed successfully.
    """
    try:
        from IPython.core import ultratb
    except ImportError:
        warnings.warn('Cannot install excepthook, IPyhon.core.ultratb not available')
        return 1
    hook = dict(color=ultratb.ColorTB, verbose=ultratb.VerboseTB).get(hook_type.lower(), None)
    if hook is None:
        return 2
    sys.excepthook = hook(**kwargs)
    return 0