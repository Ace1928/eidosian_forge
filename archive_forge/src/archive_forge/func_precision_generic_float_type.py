from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def precision_generic_float_type(self):
    """target backend will return native floating point numbers with at
        least seven decimal places when using the generic Float type.

        """
    return exclusions.open()