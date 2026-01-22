from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def identity_columns(self):
    """If a backend supports GENERATED { ALWAYS | BY DEFAULT }
        AS IDENTITY"""
    return exclusions.closed()