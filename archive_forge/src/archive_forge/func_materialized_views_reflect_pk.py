from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def materialized_views_reflect_pk(self):
    """Target database reflect MATERIALIZED VIEWs pks."""
    return exclusions.closed()