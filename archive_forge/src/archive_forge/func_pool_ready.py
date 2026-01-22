from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def pool_ready(self, event: PoolReadyEvent) -> None:
    """Abstract method to handle a :class:`PoolReadyEvent`.

        Emitted when a connection Pool is marked ready.

        :Parameters:
          - `event`: An instance of :class:`PoolReadyEvent`.

        .. versionadded:: 4.0
        """
    raise NotImplementedError