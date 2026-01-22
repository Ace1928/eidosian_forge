from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def constraint_comment_reflection(self):
    """indicates if the database support comments on constraints
        and their reflection"""
    return exclusions.closed()