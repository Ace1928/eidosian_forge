from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def uuid_data_type(self):
    """Return databases that support the UUID datatype."""
    return exclusions.closed()