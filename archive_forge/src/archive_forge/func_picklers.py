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
def picklers():
    picklers = set()
    import pickle
    picklers.add(pickle)
    for pickle_ in picklers:
        for protocol in range(-2, pickle.HIGHEST_PROTOCOL + 1):
            yield (pickle_.loads, lambda d: pickle_.dumps(d, protocol))