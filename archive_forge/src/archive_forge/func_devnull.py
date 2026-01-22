from __future__ import (absolute_import, division, print_function)
import resource
import base64
import contextlib
import errno
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
def devnull():
    """Return a file object that references devnull."""
    try:
        return devnull.file
    except AttributeError:
        devnull.file = open(os.devnull, 'w+b')
    return devnull.file