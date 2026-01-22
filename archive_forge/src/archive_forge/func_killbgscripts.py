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
@line_magic('killbgscripts')
def killbgscripts(self, _nouse_=''):
    """Kill all BG processes started by %%script and its family."""
    self.kill_bg_processes()
    print('All background processes were killed.')