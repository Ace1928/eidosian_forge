from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Type
import bson
from bson.binary import Binary
from bson.son import SON
from pymongo.errors import ConfigurationError, OperationFailure
def bson_encode(self, doc: Mapping[str, Any]) -> bytes:
    """Encode a dictionary to BSON."""
    return bson.encode(doc)