from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def group_by_complex_expression(self):
    """target platform supports SQL expressions in GROUP BY

        e.g.

        SELECT x + y AS somelabel FROM table GROUP BY x + y

        """
    return exclusions.open()