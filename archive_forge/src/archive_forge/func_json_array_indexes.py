from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def json_array_indexes(self):
    """target platform supports numeric array indexes
        within a JSON structure"""
    return self.json_type