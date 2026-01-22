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
def common_wire_version(self) -> Optional[int]:
    """Minimum of all servers' max wire versions, or None."""
    servers = self.known_servers
    if servers:
        return min((s.max_wire_version for s in self.known_servers))
    return None