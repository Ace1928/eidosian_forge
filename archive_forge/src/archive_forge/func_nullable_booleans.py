from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def nullable_booleans(self):
    """Target database allows boolean columns to store NULL."""
    return exclusions.open()