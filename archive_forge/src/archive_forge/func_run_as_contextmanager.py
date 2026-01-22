from __future__ import annotations
from collections import deque
import decimal
import gc
from itertools import chain
import random
import sys
from sys import getsizeof
import types
from . import config
from . import mock
from .. import inspect
from ..engine import Connection
from ..schema import Column
from ..schema import DropConstraint
from ..schema import DropTable
from ..schema import ForeignKeyConstraint
from ..schema import MetaData
from ..schema import Table
from ..sql import schema
from ..sql.sqltypes import Integer
from ..util import decorator
from ..util import defaultdict
from ..util import has_refcount_gc
from ..util import inspect_getfullargspec
def run_as_contextmanager(ctx, fn, *arg, **kw):
    """Run the given function under the given contextmanager,
    simulating the behavior of 'with' to support older
    Python versions.

    This is not necessary anymore as we have placed 2.6
    as minimum Python version, however some tests are still using
    this structure.

    """
    obj = ctx.__enter__()
    try:
        result = fn(obj, *arg, **kw)
        ctx.__exit__(None, None, None)
        return result
    except:
        exc_info = sys.exc_info()
        raise_ = ctx.__exit__(*exc_info)
        if not raise_:
            raise
        else:
            return raise_