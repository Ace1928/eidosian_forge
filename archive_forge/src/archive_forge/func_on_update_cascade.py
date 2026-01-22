from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def on_update_cascade(self):
    """target database must support ON UPDATE..CASCADE behavior in
        foreign keys."""
    return exclusions.open()