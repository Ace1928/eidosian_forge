from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def implements_get_lastrowid(self):
    """target dialect implements the executioncontext.get_lastrowid()
        method without reliance on RETURNING.

        """
    return exclusions.open()