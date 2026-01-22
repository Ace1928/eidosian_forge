from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def nested_aggregates(self):
    """target database can select an aggregate from a subquery that's
        also using an aggregate

        """
    return exclusions.open()