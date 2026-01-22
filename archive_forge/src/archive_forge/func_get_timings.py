from __future__ import absolute_import
import itertools
from time import time
from . import Errors
from . import DebugFlags
from . import Options
from .Errors import CompileError, InternalError, AbortError
from . import Naming
def get_timings():
    try:
        return threadlocal.cython_pipeline_timings
    except AttributeError:
        return {}