from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def sqlalchemy2_stubs(self):

    def check(config):
        try:
            __import__('sqlalchemy-stubs.ext.mypy')
        except ImportError:
            return False
        else:
            return True
    return exclusions.only_if(check)