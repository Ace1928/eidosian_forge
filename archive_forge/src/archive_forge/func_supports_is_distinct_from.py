from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def supports_is_distinct_from(self):
    """Supports some form of "x IS [NOT] DISTINCT FROM y" construct.
        Different dialects will implement their own flavour, e.g.,
        sqlite will emit "x IS NOT y" instead of "x IS DISTINCT FROM y".

        .. seealso::

            :meth:`.ColumnOperators.is_distinct_from`

        """
    return exclusions.skip_if(lambda config: not config.db.dialect.supports_is_distinct_from, "driver doesn't support an IS DISTINCT FROM construct")