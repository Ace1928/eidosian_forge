from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def publish_server_heartbeat_failed(self, connection_id: _Address, duration: float, reply: Exception, awaited: bool) -> None:
    """Publish a ServerHeartbeatFailedEvent to all server heartbeat
        listeners.

        :Parameters:
         - `connection_id`: The address (host, port) pair of the connection.
         - `duration`: The execution time of the event in the highest possible
            resolution for the platform.
         - `reply`: The command reply.
         - `awaited`: True if the response was awaited.
        """
    event = ServerHeartbeatFailedEvent(duration, reply, connection_id, awaited)
    for subscriber in self.__server_heartbeat_listeners:
        try:
            subscriber.failed(event)
        except Exception:
            _handle_exception()