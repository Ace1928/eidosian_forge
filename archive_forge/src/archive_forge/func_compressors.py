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
def compressors(self) -> Optional[list[str]]:
    return self._doc.get('compression')