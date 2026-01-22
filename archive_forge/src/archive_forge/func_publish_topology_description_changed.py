from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def publish_topology_description_changed(self, previous_description: TopologyDescription, new_description: TopologyDescription, topology_id: ObjectId) -> None:
    """Publish a TopologyDescriptionChangedEvent to all topology listeners.

        :Parameters:
         - `previous_description`: The previous topology description.
         - `new_description`: The new topology description.
         - `topology_id`: A unique identifier for the topology this server
           is a part of.
        """
    event = TopologyDescriptionChangedEvent(previous_description, new_description, topology_id)
    for subscriber in self.__topology_listeners:
        try:
            subscriber.description_changed(event)
        except Exception:
            _handle_exception()