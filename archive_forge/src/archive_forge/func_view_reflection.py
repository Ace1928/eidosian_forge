from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def view_reflection(self):
    """target database must support inspection of the full CREATE VIEW
        definition."""
    return self.views