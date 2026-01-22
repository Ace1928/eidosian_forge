from collections import namedtuple
import os
import re
import subprocess
from textwrap import dedent
from numba import config
def check_python(self):
    cmd = 'python from __future__ import print_function; import sys; print(sys.version_info[:2])'
    return self._run_cmd((cmd,))