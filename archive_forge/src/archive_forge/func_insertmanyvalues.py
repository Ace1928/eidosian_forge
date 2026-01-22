from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def insertmanyvalues(self):
    return exclusions.only_if(lambda config: config.db.dialect.supports_multivalues_insert and config.db.dialect.insert_returning and config.db.dialect.use_insertmanyvalues, "%(database)s %(does_support)s 'insertmanyvalues functionality")