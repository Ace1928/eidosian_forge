from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def computed_columns_reflect_persisted(self):
    """If persistence information is returned by the reflection of
        computed columns"""
    return exclusions.closed()