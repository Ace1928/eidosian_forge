from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
def is_python_frame(self):
    """Is this a _PyEval_EvalFrameDefault frame, or some other important
        frame? (see is_other_python_frame for what "important" means in this
        context)"""
    if self.is_evalframe():
        return True
    if self.is_other_python_frame():
        return True
    return False