from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def description_changed(self, event: ServerDescriptionChangedEvent) -> None:
    """Abstract method to handle a `ServerDescriptionChangedEvent`.

        :Parameters:
          - `event`: An instance of :class:`ServerDescriptionChangedEvent`.
        """
    raise NotImplementedError