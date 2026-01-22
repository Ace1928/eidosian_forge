from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def python311(self):
    return exclusions.only_if(lambda: util.py311, 'Python 3.11 or above required')