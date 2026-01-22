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
def jupyter_data_dir() -> str:
    """Get the config directory for Jupyter data files for this platform and user.

    These are non-transient, non-configuration files.

    Returns JUPYTER_DATA_DIR if defined, else a platform-appropriate path.
    """
    env = os.environ
    if env.get('JUPYTER_DATA_DIR'):
        return env['JUPYTER_DATA_DIR']
    if use_platform_dirs():
        return platformdirs.user_data_dir(APPNAME, appauthor=False)
    home = get_home_dir()
    if sys.platform == 'darwin':
        return str(Path(home, 'Library', 'Jupyter'))
    if sys.platform == 'win32':
        appdata = os.environ.get('APPDATA', None)
        if appdata:
            return str(Path(appdata, 'jupyter').resolve())
        return pjoin(jupyter_config_dir(), 'data')
    xdg = env.get('XDG_DATA_HOME', None)
    if not xdg:
        xdg = pjoin(home, '.local', 'share')
    return pjoin(xdg, 'jupyter')