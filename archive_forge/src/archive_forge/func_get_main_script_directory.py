from __future__ import annotations
import contextlib
import errno
import io
import os
from pathlib import Path
from streamlit import env_util, util
from streamlit.string_util import is_binary_string
def get_main_script_directory(main_script):
    """Return the full path to the main script directory.

    Parameters
    ----------
    main_script : str
        The main script path. The path can be an absolute path or a relative
        path.

    Returns
    -------
    str
        The full path to the main script directory.
    """
    main_script_path = normalize_path_join(os.getcwd(), main_script)
    return os.path.dirname(main_script_path)