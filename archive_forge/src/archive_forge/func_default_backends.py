import os
import sys
from contextlib import suppress
from typing import (
from .file import GitFile
@classmethod
def default_backends(cls) -> List[ConfigFile]:
    """Retrieve the default configuration.

        See git-config(1) for details on the files searched.
        """
    paths = []
    paths.append(os.path.expanduser('~/.gitconfig'))
    paths.append(get_xdg_config_home_path('git', 'config'))
    if 'GIT_CONFIG_NOSYSTEM' not in os.environ:
        paths.append('/etc/gitconfig')
        if sys.platform == 'win32':
            paths.extend(get_win_system_paths())
    backends = []
    for path in paths:
        try:
            cf = ConfigFile.from_path(path)
        except FileNotFoundError:
            continue
        backends.append(cf)
    return backends