from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def pool_cleared(self, event: PoolClearedEvent) -> None:
    """Abstract method to handle a `PoolClearedEvent`.

        Emitted when a connection Pool is cleared.

        :Parameters:
          - `event`: An instance of :class:`PoolClearedEvent`.
        """
    raise NotImplementedError