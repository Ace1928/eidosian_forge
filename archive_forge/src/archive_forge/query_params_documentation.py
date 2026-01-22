from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Iterator, MutableMapping
from urllib import parse
from streamlit.constants import EMBED_QUERY_PARAMS_KEYS
from streamlit.errors import StreamlitAPIException
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
Retrieves a value for a given key in query parameters.
        Returns the last item in a list or an empty string if empty.
        If the key is not present, raise KeyError.
        