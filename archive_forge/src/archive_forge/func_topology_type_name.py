from __future__ import annotations
from random import sample
from typing import (
from bson.min_key import MinKey
from bson.objectid import ObjectId
from pymongo import common
from pymongo.errors import ConfigurationError
from pymongo.read_preferences import ReadPreference, _AggWritePref, _ServerMode
from pymongo.server_description import ServerDescription
from pymongo.server_selectors import Selection
from pymongo.server_type import SERVER_TYPE
from pymongo.typings import _Address
@property
def topology_type_name(self) -> str:
    """The topology type as a human readable string.

        .. versionadded:: 3.4
        """
    return TOPOLOGY_TYPE._fields[self._topology_type]