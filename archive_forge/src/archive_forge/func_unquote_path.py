import os
import shlex
import sys
from pbr import find_package
from pbr.hooks import base
def unquote_path(path):
    if os.name == 'nt':
        path = path.replace('\\', '/')
        return ''.join(shlex.split(path)).replace('/', '\\')
    return ''.join(shlex.split(path))