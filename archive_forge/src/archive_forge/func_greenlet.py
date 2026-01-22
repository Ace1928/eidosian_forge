from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def greenlet(self):

    def go(config):
        if not _test_asyncio.ENABLE_ASYNCIO:
            return False
        try:
            import greenlet
        except ImportError:
            return False
        else:
            return True
    return exclusions.only_if(go)