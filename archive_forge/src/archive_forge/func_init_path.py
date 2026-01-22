import glob
from itertools import chain
import os
import sys
from traitlets.config.application import boolean_flag
from traitlets.config.configurable import Configurable
from traitlets.config.loader import Config
from IPython.core.application import SYSTEM_CONFIG_DIRS, ENV_CONFIG_DIRS
from IPython.core import pylabtools
from IPython.utils.contexts import preserve_keys
from IPython.utils.path import filefind
from traitlets import (
from IPython.terminal import pt_inputhooks
def init_path(self):
    """Add current working directory, '', to sys.path

        Unlike Python's default, we insert before the first `site-packages`
        or `dist-packages` directory,
        so that it is after the standard library.

        .. versionchanged:: 7.2
            Try to insert after the standard library, instead of first.
        .. versionchanged:: 8.0
            Allow optionally not including the current directory in sys.path
        """
    if '' in sys.path or self.ignore_cwd:
        return
    for idx, path in enumerate(sys.path):
        parent, last_part = os.path.split(path)
        if last_part in {'site-packages', 'dist-packages'}:
            break
    else:
        idx = 0
    sys.path.insert(idx, '')