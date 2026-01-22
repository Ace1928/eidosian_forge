from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def standalone_null_binds_whereclause(self):
    """target database/driver supports bound parameters with NULL in the
        WHERE clause, in situations where it has to be typed.

        """
    return exclusions.open()