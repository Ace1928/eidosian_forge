from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def precision_numerics_enotation_small(self):
    """target backend supports Decimal() objects using E notation
        to represent very small values."""
    return exclusions.closed()