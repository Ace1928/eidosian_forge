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
def use_fast_gil_utility_code(self):
    if self.globalstate.directives['fast_gil']:
        self.globalstate.use_utility_code(UtilityCode.load_cached('FastGil', 'ModuleSetupCode.c'))
    else:
        self.globalstate.use_utility_code(UtilityCode.load_cached('NoFastGil', 'ModuleSetupCode.c'))