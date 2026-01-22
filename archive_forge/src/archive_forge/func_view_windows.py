import os
import io
import re
import sys
import errno
import platform
import subprocess
import contextlib
from ._compat import stderr_write_binary
from . import tools
@tools.attach(view, 'windows')
def view_windows(filepath):
    """Start filepath with its associated application (windows)."""
    os.startfile(os.path.normpath(filepath))