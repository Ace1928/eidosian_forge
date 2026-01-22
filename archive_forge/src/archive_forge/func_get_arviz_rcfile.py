import locale
import logging
import os
import pprint
import re
import sys
import warnings
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict
from typing_extensions import Literal
import numpy as np
def get_arviz_rcfile():
    """Get arvizrc file.

    The file location is determined in the following order:

    - ``$PWD/arvizrc``
    - ``$ARVIZ_DATA/arvizrc``
    - On Linux,
        - ``$XDG_CONFIG_HOME/arviz/arvizrc`` (if ``$XDG_CONFIG_HOME``
          is defined)
        - or ``$HOME/.config/arviz/arvizrc`` (if ``$XDG_CONFIG_HOME``
          is not defined)
    - On other platforms,
        - ``$HOME/.arviz/arvizrc`` if ``$HOME`` is defined

    Otherwise, the default defined in ``rcparams.py`` file will be used.
    """

    def gen_candidates():
        yield os.path.join(os.getcwd(), 'arvizrc')
        arviz_data_dir = os.environ.get('ARVIZ_DATA')
        if arviz_data_dir:
            yield os.path.join(arviz_data_dir, 'arvizrc')
        xdg_base = os.environ.get('XDG_CONFIG_HOME', str(Path.home() / '.config'))
        if sys.platform.startswith(('linux', 'freebsd')):
            configdir = str(Path(xdg_base, 'arviz'))
        else:
            configdir = str(Path.home() / '.arviz')
        yield os.path.join(configdir, 'arvizrc')
    for fname in gen_candidates():
        if os.path.exists(fname) and (not os.path.isdir(fname)):
            return fname
    return None