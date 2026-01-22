from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def unique_index_reflect_as_unique_constraints(self):
    """Target database reflects unique indexes as unique constrains."""
    return exclusions.closed()