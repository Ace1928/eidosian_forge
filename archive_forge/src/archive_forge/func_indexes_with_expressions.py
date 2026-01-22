from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def indexes_with_expressions(self):
    """target database supports CREATE INDEX against SQL expressions."""
    return exclusions.closed()