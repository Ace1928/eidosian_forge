from __future__ import absolute_import, division, print_function
import os
import re
from contextlib import contextmanager
from struct import Struct
from ansible.module_utils.six import PY3
def secure_write(path, mode, content):
    with secure_open(path, mode) as fd:
        os.write(fd, content)