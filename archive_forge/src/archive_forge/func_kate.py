import os
import shlex
import subprocess
import sys
from IPython import get_ipython
from IPython.core.error import TryNext
from IPython.utils import py3compat
def kate(exe=u'kate'):
    install_editor(exe + u' -u -l {line} {filename}')