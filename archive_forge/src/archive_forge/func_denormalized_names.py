from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def denormalized_names(self):
    """Target database must have 'denormalized', i.e.
        UPPERCASE as case insensitive names."""
    return exclusions.skip_if(lambda config: not config.db.dialect.requires_name_normalize, 'Backend does not require denormalized names.')