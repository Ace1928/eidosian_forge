from __future__ import annotations
import shutil
import tempfile
from streamlit import util
Temporary directory context manager.

    Creates a temporary directory that exists within the context manager scope.
    It returns the path to the created directory.
    Wrapper on top of tempfile.mkdtemp.

    Parameters
    ----------
    suffix : str or None
        Suffix to the filename.
    prefix : str or None
        Prefix to the filename.
    dir : str or None
        Enclosing directory.

    