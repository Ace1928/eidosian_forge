from __future__ import annotations
import math
from datetime import timedelta
from typing import Any, Literal, overload
from streamlit import config
from streamlit.errors import MarkdownFormattedException, StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.forward_msg_cache import populate_hash_if_needed
def serialize_forward_msg(msg: ForwardMsg) -> bytes:
    """Serialize a ForwardMsg to send to a client.

    If the message is too large, it will be converted to an exception message
    instead.
    """
    populate_hash_if_needed(msg)
    msg_str = msg.SerializeToString()
    if len(msg_str) > get_max_message_size_bytes():
        import streamlit.elements.exception as exception
        exception.marshall(msg.delta.new_element.exception, MessageSizeError(msg_str))
        msg_str = msg.SerializeToString()
    return msg_str