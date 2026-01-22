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
def enq(self, group):
    depth = group.depth
    while depth > len(self.queue) - 1:
        self.queue.append([])
    self.queue[depth].append(group)