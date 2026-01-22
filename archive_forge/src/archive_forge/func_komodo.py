import os
import shlex
import subprocess
import sys
from IPython import get_ipython
from IPython.core.error import TryNext
from IPython.utils import py3compat
def komodo(exe=u'komodo'):
    """ Activestate Komodo [Edit] """
    install_editor(exe + u' -l {line} {filename}', wait=True)