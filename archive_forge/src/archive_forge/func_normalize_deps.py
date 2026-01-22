from __future__ import absolute_import
import itertools
from time import time
from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming
def normalize_deps(utilcodes):
    deps = {utilcode: utilcode for utilcode in utilcodes}
    for utilcode in utilcodes:
        utilcode.requires = [deps.setdefault(dep, dep) for dep in utilcode.requires or ()]