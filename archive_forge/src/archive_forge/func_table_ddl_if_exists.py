from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def table_ddl_if_exists(self):
    """target platform supports IF NOT EXISTS / IF EXISTS for tables."""
    return exclusions.closed()