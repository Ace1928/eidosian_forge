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
def has_readable_server(self, read_preference: _ServerMode=ReadPreference.PRIMARY) -> bool:
    """Does this topology have any readable servers available matching the
        given read preference?

        :Parameters:
          - `read_preference`: an instance of a read preference from
            :mod:`~pymongo.read_preferences`. Defaults to
            :attr:`~pymongo.read_preferences.ReadPreference.PRIMARY`.

        .. note:: When connected directly to a single server this method
          always returns ``True``.

        .. versionadded:: 3.4
        """
    common.validate_read_preference('read_preference', read_preference)
    return any(self.apply_selector(read_preference))