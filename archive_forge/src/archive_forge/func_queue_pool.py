from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def queue_pool(self):
    """target database is using QueuePool"""

    def go(config):
        return isinstance(config.db.pool, QueuePool)
    return exclusions.only_if(go)