from __future__ import annotations
from typing import TYPE_CHECKING, cast
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Toast_pb2 import Toast as ToastProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.string_util import clean_text, validate_emoji
from streamlit.type_util import SupportsStr
def validate_text(toast_text: SupportsStr) -> SupportsStr:
    if str(toast_text) == '':
        raise StreamlitAPIException(f'Toast body cannot be blank - please provide a message.')
    else:
        return toast_text