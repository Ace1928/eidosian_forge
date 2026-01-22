import contextlib
import logging
import os
from pathlib import Path
import sys
import warnings
import matplotlib as mpl
from matplotlib import _api, _docstring, _rc_params_in_file, rcParamsDefault
def reload_library():
    """Reload the style library."""
    library.clear()
    library.update(update_user_library(_base_library))
    available[:] = sorted(library.keys())