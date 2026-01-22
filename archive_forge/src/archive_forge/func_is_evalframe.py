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
def is_evalframe(self):
    """Is this a _PyEval_EvalFrameDefault frame?"""
    if self._gdbframe.name() == EVALFRAME:
        '\n            I believe we also need to filter on the inline\n            struct frame_id.inline_depth, only regarding frames with\n            an inline depth of 0 as actually being this function\n\n            So we reject those with type gdb.INLINE_FRAME\n            '
        if self._gdbframe.type() == gdb.NORMAL_FRAME:
            return True
    return False