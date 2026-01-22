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
def do_next(self, arg):
    """n(ext)
        Continue execution until the next line in the current function
        is reached or it returns.
        """
    self.set_next(self.curframe)
    return 1