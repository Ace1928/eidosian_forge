import asyncio
import asyncio.exceptions
import atexit
import errno
import os
import signal
import sys
import time
from subprocess import CalledProcessError
from threading import Thread
from traitlets import Any, Dict, List, default
from IPython.core import magic_arguments
from IPython.core.async_helpers import _AsyncIOProxy
from IPython.core.magic import Magics, cell_magic, line_magic, magics_class
from IPython.utils.process import arg_split
@magic_arguments.magic_arguments()
@script_args
def named_script_magic(line, cell):
    if line:
        line = '%s %s' % (script, line)
    else:
        line = script
    return self.shebang(line, cell)