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
def put_setup_refcount_context(self, name, acquire_gil=False):
    name = name.as_c_string_literal()
    if acquire_gil:
        self.globalstate.use_utility_code(UtilityCode.load_cached('ForceInitThreads', 'ModuleSetupCode.c'))
    self.putln('__Pyx_RefNannySetupContext(%s, %d);' % (name, acquire_gil and 1 or 0))