from __future__ import with_statement
import os
import sys
import textwrap
import unittest
import subprocess
import tempfile
def open_temp_file():
    if sys.version_info >= (2, 6):
        file = tempfile.NamedTemporaryFile(delete=False)
        filename = file.name
    else:
        fd, filename = tempfile.mkstemp()
        file = os.fdopen(fd, 'w+b')
    return (file, filename)