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
def put_release_ensured_gil(self, variable=None):
    """
        Releases the GIL, corresponds to `put_ensure_gil`.
        """
    self.use_fast_gil_utility_code()
    if not variable:
        variable = '__pyx_gilstate_save'
    self.putln('#ifdef WITH_THREAD')
    self.putln('__Pyx_PyGILState_Release(%s);' % variable)
    self.putln('#endif')