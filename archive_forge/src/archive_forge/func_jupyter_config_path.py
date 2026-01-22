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
def jupyter_config_path() -> list[str]:
    """Return the search path for Jupyter config files as a list.

    If the JUPYTER_PREFER_ENV_PATH environment variable is set, the
    environment-level directories will have priority over user-level
    directories.

    If the Python site.ENABLE_USER_SITE variable is True, we also add the
    appropriate Python user site subdirectory to the user-level directories.
    """
    if os.environ.get('JUPYTER_NO_CONFIG'):
        return [jupyter_config_dir()]
    paths: list[str] = []
    if os.environ.get('JUPYTER_CONFIG_PATH'):
        paths.extend((p.rstrip(os.sep) for p in os.environ['JUPYTER_CONFIG_PATH'].split(os.pathsep)))
    user = [jupyter_config_dir()]
    if site.ENABLE_USER_SITE:
        userbase: Optional[str]
        userbase = site.getuserbase() if hasattr(site, 'getuserbase') else site.USER_BASE
        if userbase:
            userdir = str(Path(userbase, 'etc', 'jupyter'))
            if userdir not in user:
                user.append(userdir)
    env = [p for p in ENV_CONFIG_PATH if p not in SYSTEM_CONFIG_PATH]
    if prefer_environment_over_user():
        paths.extend(env)
        paths.extend(user)
    else:
        paths.extend(user)
        paths.extend(env)
    paths.extend(SYSTEM_CONFIG_PATH)
    return paths