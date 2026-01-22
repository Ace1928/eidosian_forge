from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def no_greenlet(self):

    def go(config):
        try:
            import greenlet
        except ImportError:
            return True
        else:
            return False
    return exclusions.only_if(go)