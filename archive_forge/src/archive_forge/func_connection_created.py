from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def connection_created(self, event: ConnectionCreatedEvent) -> None:
    """Abstract method to handle a :class:`ConnectionCreatedEvent`.

        Emitted when a connection Pool creates a Connection object.

        :Parameters:
          - `event`: An instance of :class:`ConnectionCreatedEvent`.
        """
    raise NotImplementedError