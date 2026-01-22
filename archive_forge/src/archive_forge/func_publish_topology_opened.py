from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def publish_topology_opened(self, topology_id: ObjectId) -> None:
    """Publish a TopologyOpenedEvent to all topology listeners.

        :Parameters:
         - `topology_id`: A unique identifier for the topology this server
           is a part of.
        """
    event = TopologyOpenedEvent(topology_id)
    for subscriber in self.__topology_listeners:
        try:
            subscriber.opened(event)
        except Exception:
            _handle_exception()