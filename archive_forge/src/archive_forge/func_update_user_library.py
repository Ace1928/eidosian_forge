import contextlib
import logging
import os
from pathlib import Path
import sys
import warnings
import matplotlib as mpl
from matplotlib import _api, _docstring, _rc_params_in_file, rcParamsDefault
def update_user_library(library):
    """Update style library with user-defined rc files."""
    for stylelib_path in map(os.path.expanduser, USER_LIBRARY_PATHS):
        styles = read_style_directory(stylelib_path)
        update_nested_dict(library, styles)
    return library