from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Iterator, MutableMapping
from urllib import parse
from streamlit.constants import EMBED_QUERY_PARAMS_KEYS
from streamlit.errors import StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
def missing_key_error_message(key: str) -> str:
    return f'st.query_params has no key "{key}". Did you forget to initialize it?'