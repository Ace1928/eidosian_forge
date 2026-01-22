import dataclasses
import glob as py_glob
import io
import os
import os.path
import sys
import tempfile
from tensorboard.compat.tensorflow_stub import compat, errors
def register_filesystem(prefix, filesystem):
    if ':' in prefix:
        raise ValueError('Filesystem prefix cannot contain a :')
    _REGISTERED_FILESYSTEMS[prefix] = filesystem