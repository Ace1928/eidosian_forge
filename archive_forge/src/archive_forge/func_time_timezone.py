from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def time_timezone(self):
    """target dialect supports representation of Python
        datetime.time() with tzinfo with Time(timezone=True)."""
    return exclusions.closed()