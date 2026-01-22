from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def empty_inserts(self):
    """target platform supports INSERT with no values, i.e.
        INSERT DEFAULT VALUES or equivalent."""
    return exclusions.only_if(lambda config: config.db.dialect.supports_empty_insert or config.db.dialect.supports_default_values or config.db.dialect.supports_default_metavalue, 'empty inserts not supported')