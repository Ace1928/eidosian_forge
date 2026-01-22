from __future__ import absolute_import
import cython
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
def put_acquire_gil(self, variable=None, unknown_gil_state=True):
    """
        Acquire the GIL. The thread's thread state must have been initialized
        by a previous `put_release_gil`
        """
    self.use_fast_gil_utility_code()
    self.putln('#ifdef WITH_THREAD')
    self.putln('__Pyx_FastGIL_Forget();')
    if variable:
        self.putln('_save = %s;' % variable)
    if unknown_gil_state:
        self.putln('if (_save) {')
    self.putln('Py_BLOCK_THREADS')
    if unknown_gil_state:
        self.putln('}')
    self.putln('#endif')