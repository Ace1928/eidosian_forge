from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def publish_connection_checked_in(self, address: _Address, connection_id: int) -> None:
    """Publish a :class:`ConnectionCheckedInEvent` to all connection
        listeners.
        """
    event = ConnectionCheckedInEvent(address, connection_id)
    for subscriber in self.__cmap_listeners:
        try:
            subscriber.connection_checked_in(event)
        except Exception:
            _handle_exception()