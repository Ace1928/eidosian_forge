import os
import shlex
import subprocess
import sys
from IPython import get_ipython
from IPython.core.error import TryNext
from IPython.utils import py3compat
def call_editor(self, filename, line=0):
    if line is None:
        line = 0
    cmd = template.format(filename=shlex.quote(filename), line=line)
    print('>', cmd)
    if sys.platform.startswith('win'):
        cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, shell=True)
    if proc.wait() != 0:
        raise TryNext()
    if wait:
        py3compat.input('Press Enter when done editing:')