from __future__ import annotations
import logging # isort:skip
import json
from typing import (
import bokeh.util.serialization as bkserial
from ..core.json_encoder import serialize_json
from ..core.serialization import Buffer, Serialized
from ..core.types import ID
from .exceptions import MessageError, ProtocolError
@classmethod
def create_header(cls, request_id: ID | None=None) -> Header:
    """ Return a message header fragment dict.

        Args:
            request_id (str or None) :
                Message ID of the message this message replies to

        Returns:
            dict : a message header

        """
    header = Header(msgid=bkserial.make_id(), msgtype=cls.msgtype)
    if request_id is not None:
        header['reqid'] = request_id
    return header