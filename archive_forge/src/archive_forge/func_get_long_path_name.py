import os
import sys
import errno
import shutil
import random
import glob
import warnings
from IPython.utils.process import system
def get_long_path_name(path):
    """Expand a path into its long form.

    On Windows this expands any ~ in the paths. On other platforms, it is
    a null operation.
    """
    return _get_long_path_name(path)