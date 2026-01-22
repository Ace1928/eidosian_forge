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
def close_global_decls(self):
    self.generate_const_declarations()
    if Options.cache_builtins:
        w = self.parts['cached_builtins']
        w.putln('return 0;')
        if w.label_used(w.error_label):
            w.put_label(w.error_label)
            w.putln('return -1;')
        w.putln('}')
        w.exit_cfunc_scope()
    w = self.parts['cached_constants']
    w.put_finish_refcount_context()
    w.putln('return 0;')
    if w.label_used(w.error_label):
        w.put_label(w.error_label)
        w.put_finish_refcount_context()
        w.putln('return -1;')
    w.putln('}')
    w.exit_cfunc_scope()
    for part in ['init_globals', 'init_constants']:
        w = self.parts[part]
        w.putln('return 0;')
        if w.label_used(w.error_label):
            w.put_label(w.error_label)
            w.putln('return -1;')
        w.putln('}')
        w.exit_cfunc_scope()
    if Options.generate_cleanup_code:
        w = self.parts['cleanup_globals']
        w.putln('}')
        w.exit_cfunc_scope()
    if Options.generate_cleanup_code:
        w = self.parts['cleanup_module']
        w.putln('}')
        w.exit_cfunc_scope()