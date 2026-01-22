from __future__ import annotations
import copy
import datetime
import itertools
from typing import Any, Generic, Mapping, Optional
from bson.objectid import ObjectId
from pymongo import common
from pymongo.server_type import SERVER_TYPE
from pymongo.typings import ClusterTime, _DocumentType
@property
def speculative_authenticate(self) -> Optional[Mapping[str, Any]]:
    """The speculativeAuthenticate field."""
    return self._doc.get('speculativeAuthenticate')