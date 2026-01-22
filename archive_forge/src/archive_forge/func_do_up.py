import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
def do_up(self, arg):
    """u(p) [count]
        Move the current frame count (default one) levels up in the
        stack trace (to an older frame).
        """
    if self.curindex == 0:
        self.error('Oldest frame')
        return
    try:
        count = int(arg or 1)
    except ValueError:
        self.error('Invalid frame count (%s)' % arg)
        return
    if count < 0:
        newframe = 0
    else:
        newframe = max(0, self.curindex - count)
    self._select_frame(newframe)