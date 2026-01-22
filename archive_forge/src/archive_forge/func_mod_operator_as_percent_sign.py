from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def mod_operator_as_percent_sign(self):
    """target database must use a plain percent '%' as the 'modulus'
        operator."""
    return exclusions.closed()