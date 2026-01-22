from __future__ import annotations
import datetime
import random
import struct
from io import BytesIO as _BytesIO
from typing import (
import bson
from bson import CodecOptions, _decode_selective, _dict_to_bson, _make_c_string, encode
from bson.int64 import Int64
from bson.raw_bson import (
from bson.son import SON
from pymongo.errors import (
from pymongo.hello import HelloCompat
from pymongo.helpers import _handle_reauth
from pymongo.read_preferences import ReadPreference
from pymongo.write_concern import WriteConcern
def unack_write(self, cmd: MutableMapping[str, Any], request_id: int, msg: bytes, max_doc_size: int, docs: list[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """A proxy for Connection.unack_write that handles event publishing."""
    if self.publish:
        assert self.start_time is not None
        duration = datetime.datetime.now() - self.start_time
        cmd = self._start(cmd, request_id, docs)
        start = datetime.datetime.now()
    try:
        result = self.conn.unack_write(msg, max_doc_size)
        if self.publish:
            duration = datetime.datetime.now() - start + duration
            if result is not None:
                reply = _convert_write_result(self.name, cmd, result)
            else:
                reply = {'ok': 1}
            self._succeed(request_id, reply, duration)
    except Exception as exc:
        if self.publish:
            assert self.start_time is not None
            duration = datetime.datetime.now() - start + duration
            if isinstance(exc, OperationFailure):
                failure: _DocumentOut = _convert_write_result(self.name, cmd, exc.details)
            elif isinstance(exc, NotPrimaryError):
                failure = exc.details
            else:
                failure = _convert_exception(exc)
            self._fail(request_id, failure, duration)
        raise
    finally:
        self.start_time = datetime.datetime.now()
    return result