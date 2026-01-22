from contextlib import contextmanager
import datetime
import os
import re
import sys
import types
from collections import deque
from inspect import signature
from io import StringIO
from warnings import warn
from IPython.utils.decorators import undoc
from IPython.utils.py3compat import PYPY
from typing import Dict
def break_(self):
    """
        Explicitly insert a newline into the output, maintaining correct indentation.
        """
    group = self.group_queue.deq()
    if group:
        self._break_one_group(group)
    self.flush()
    self.output.write(self.newline)
    self.output.write(' ' * self.indentation)
    self.output_width = self.indentation
    self.buffer_width = 0