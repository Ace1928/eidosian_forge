from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def dbapi_lastrowid(self):
    """target platform includes a 'lastrowid' accessor on the DBAPI
        cursor object.

        """
    return exclusions.closed()