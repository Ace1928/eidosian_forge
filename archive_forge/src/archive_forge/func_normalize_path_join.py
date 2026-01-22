from __future__ import annotations
import contextlib
import errno
import io
import os
from pathlib import Path
from streamlit import env_util, util
from streamlit.string_util import is_binary_string
def normalize_path_join(*args):
    """Return the normalized path of the joined path.

    Parameters
    ----------
    *args : str
        The path components to join.

    Returns
    -------
    str
        The normalized path of the joined path.
    """
    return os.path.normpath(os.path.join(*args))