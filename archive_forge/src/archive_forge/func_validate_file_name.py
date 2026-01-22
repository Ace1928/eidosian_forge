import os
import pathlib
from django.core.exceptions import SuspiciousFileOperation
def validate_file_name(name, allow_relative_path=False):
    if os.path.basename(name) in {'', '.', '..'}:
        raise SuspiciousFileOperation("Could not derive file name from '%s'" % name)
    if allow_relative_path:
        path = pathlib.PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:
            raise SuspiciousFileOperation("Detected path traversal attempt in '%s'" % name)
    elif name != os.path.basename(name):
        raise SuspiciousFileOperation("File name '%s' includes path elements" % name)
    return name