import os
import sys
from os.path import pardir, realpath
def get_config_h_filename():
    """Return the path of pyconfig.h."""
    if _PYTHON_BUILD:
        if os.name == 'nt':
            inc_dir = os.path.join(_PROJECT_BASE, 'PC')
        else:
            inc_dir = _PROJECT_BASE
    else:
        inc_dir = get_path('platinclude')
    return os.path.join(inc_dir, 'pyconfig.h')