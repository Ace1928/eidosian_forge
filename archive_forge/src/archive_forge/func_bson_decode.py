from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Type
import bson
from bson.binary import Binary
from bson.son import SON
from pymongo.errors import ConfigurationError, OperationFailure
def bson_decode(self, data: _ReadableBuffer) -> Mapping[str, Any]:
    """Decode BSON to a dictionary."""
    return bson.decode(data)