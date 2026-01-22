import io
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import (
from mypy_extensions import mypyc_attr
from packaging.specifiers import InvalidSpecifier, Specifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPatternError
from black.handle_ipynb_magics import jupyter_dependencies_are_installed
from black.mode import TargetVersion
from black.output import err
from black.report import Report
@lru_cache
def find_user_pyproject_toml() -> Path:
    """Return the path to the top-level user configuration for black.

    This looks for ~\\.black on Windows and ~/.config/black on Linux and other
    Unix systems.

    May raise:
    - RuntimeError: if the current user has no homedir
    - PermissionError: if the current process cannot access the user's homedir
    """
    if sys.platform == 'win32':
        user_config_path = Path.home() / '.black'
    else:
        config_root = os.environ.get('XDG_CONFIG_HOME', '~/.config')
        user_config_path = Path(config_root).expanduser() / 'black'
    return _cached_resolve(user_config_path)