from __future__ import annotations
import logging # isort:skip
import json
from typing import (
import bokeh.util.serialization as bkserial
from ..core.json_encoder import serialize_json
from ..core.serialization import Buffer, Serialized
from ..core.types import ID
from .exceptions import MessageError, ProtocolError
@property
def metadata_json(self) -> str:
    if not self._metadata_json:
        self._metadata_json = json.dumps(self.metadata)
    return self._metadata_json