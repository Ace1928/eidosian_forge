from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def autoincrement_insert(self):
    """target platform generates new surrogate integer primary key values
        when insert() is executed, excluding the pk column."""
    return exclusions.open()