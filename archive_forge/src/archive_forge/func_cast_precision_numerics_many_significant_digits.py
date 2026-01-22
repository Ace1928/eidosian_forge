from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def cast_precision_numerics_many_significant_digits(self):
    """same as precision_numerics_many_significant_digits but within the
        context of a CAST statement (hello MySQL)

        """
    return self.precision_numerics_many_significant_digits