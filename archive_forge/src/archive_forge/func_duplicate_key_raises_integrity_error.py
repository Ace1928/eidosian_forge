from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def duplicate_key_raises_integrity_error(self):
    """target dialect raises IntegrityError when reporting an INSERT
        with a primary key violation.  (hint: it should)

        """
    return exclusions.open()