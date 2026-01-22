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
def print_stack_trace(self):
    try:
        for frame_lineno in self.stack:
            self.print_stack_entry(frame_lineno)
    except KeyboardInterrupt:
        pass