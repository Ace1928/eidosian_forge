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
def prefer_environment_over_user() -> bool:
    """Determine if environment-level paths should take precedence over user-level paths."""
    if 'JUPYTER_PREFER_ENV_PATH' in os.environ:
        return envset('JUPYTER_PREFER_ENV_PATH')
    if sys.prefix != sys.base_prefix and _do_i_own(sys.prefix):
        return True
    if 'CONDA_PREFIX' in os.environ and sys.prefix.startswith(os.environ['CONDA_PREFIX']) and (os.environ.get('CONDA_DEFAULT_ENV', 'base') != 'base') and _do_i_own(sys.prefix):
        return True
    return False