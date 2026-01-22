from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def date_implicit_bound(self):
    """target dialect when given a date object will bind it such
        that the database server knows the object is a date, and not
        a plain string.

        """
    return exclusions.open()