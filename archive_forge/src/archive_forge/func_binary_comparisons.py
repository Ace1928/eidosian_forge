from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def binary_comparisons(self):
    """target database/driver can allow BLOB/BINARY fields to be compared
        against a bound parameter value.
        """
    return exclusions.open()