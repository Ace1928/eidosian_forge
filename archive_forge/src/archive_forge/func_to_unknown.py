from __future__ import annotations
import time
import warnings
from typing import Any, Mapping, Optional
from bson import EPOCH_NAIVE
from bson.objectid import ObjectId
from pymongo.hello import Hello
from pymongo.server_type import SERVER_TYPE
from pymongo.typings import ClusterTime, _Address
def to_unknown(self, error: Optional[Exception]=None) -> ServerDescription:
    unknown = ServerDescription(self.address, error=error)
    unknown._topology_version = self.topology_version
    return unknown