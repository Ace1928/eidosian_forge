from asyncio import iscoroutinefunction
from contextlib import contextmanager
from functools import partial, wraps
from types import coroutine
import builtins
import inspect
import linecache
import logging
import os
import io
import pdb
import subprocess
import sys
import time
import traceback
import warnings
import psutil
@contextmanager
def call_on_stack(self, func, *args, **kwds):
    self.current_stack_level += 1
    self.stack[func].append(self.current_stack_level)
    yield func(*args, **kwds)
    self.current_stack_level -= 1