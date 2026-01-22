from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def generic_classes(self):
    """If X[Y] can be implemented with ``__class_getitem__``. py3.7+"""
    return exclusions.open()