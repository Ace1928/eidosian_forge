from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def server_side_cursors(self):
    """Target dialect must support server side cursors."""
    return exclusions.only_if([lambda config: config.db.dialect.supports_server_side_cursors], 'no server side cursors support')