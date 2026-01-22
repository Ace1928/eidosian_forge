from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def duplicate_names_in_cursor_description(self):
    """target platform supports a SELECT statement that has
        the same name repeated more than once in the columns list."""
    return exclusions.open()