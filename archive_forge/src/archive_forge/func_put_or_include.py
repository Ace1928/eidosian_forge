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
def put_or_include(self, code, name):
    include_dir = self.globalstate.common_utility_include_dir
    if include_dir and len(code) > 1024:
        include_file = '%s_%s.h' % (name, hashlib.sha1(code.encode('utf8')).hexdigest())
        path = os.path.join(include_dir, include_file)
        if not os.path.exists(path):
            tmp_path = '%s.tmp%s' % (path, os.getpid())
            with closing(Utils.open_new_file(tmp_path)) as f:
                f.write(code)
            shutil.move(tmp_path, path)
        code = '#include "%s"\n' % path
    self.put(code)