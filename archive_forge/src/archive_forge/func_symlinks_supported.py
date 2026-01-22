import os
import tempfile
from os.path import abspath, dirname, join, normcase, sep
from pathlib import Path
from django.core.exceptions import SuspiciousFileOperation
def symlinks_supported():
    """
    Return whether or not creating symlinks are supported in the host platform
    and/or if they are allowed to be created (e.g. on Windows it requires admin
    permissions).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        original_path = os.path.join(temp_dir, 'original')
        symlink_path = os.path.join(temp_dir, 'symlink')
        os.makedirs(original_path)
        try:
            os.symlink(original_path, symlink_path)
            supported = True
        except (OSError, NotImplementedError):
            supported = False
        return supported