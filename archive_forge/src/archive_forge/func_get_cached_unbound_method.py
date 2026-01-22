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
def get_cached_unbound_method(self, type_cname, method_name):
    key = (type_cname, method_name)
    try:
        cname = self.cached_cmethods[key]
    except KeyError:
        cname = self.cached_cmethods[key] = self.new_const_cname('umethod', '%s_%s' % (type_cname, method_name))
    return cname