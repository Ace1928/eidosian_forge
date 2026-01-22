import os
import sys
import warnings
from typing import ClassVar, Set
def ensure_dir_exists(dirname):
    """Ensure a directory exists, creating if necessary."""
    try:
        os.makedirs(dirname)
    except FileExistsError:
        pass