from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
@property
def awaited(self) -> bool:
    """Whether the heartbeat was awaited.

        If true, then :meth:`duration` reflects the sum of the round trip time
        to the server and the time that the server waited before sending a
        response.

        .. versionadded:: 3.11
        """
    return super().awaited