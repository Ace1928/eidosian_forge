from __future__ import annotations
import math
from datetime import timedelta
from typing import Any, Literal, overload
from streamlit import config
from streamlit.errors import MarkdownFormattedException, StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.forward_msg_cache import populate_hash_if_needed
Returns the max websocket message size in bytes.

    This will lazyload the value from the config and store it in the global symbol table.
    