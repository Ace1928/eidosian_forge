from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import (
from google.protobuf.message import Message
from typing_extensions import TypeAlias
from streamlit import config, util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow
from streamlit.proto.Button_pb2 import Button
from streamlit.proto.CameraInput_pb2 import CameraInput
from streamlit.proto.ChatInput_pb2 import ChatInput
from streamlit.proto.Checkbox_pb2 import Checkbox
from streamlit.proto.ColorPicker_pb2 import ColorPicker
from streamlit.proto.Components_pb2 import ComponentInstance
from streamlit.proto.DateInput_pb2 import DateInput
from streamlit.proto.DownloadButton_pb2 import DownloadButton
from streamlit.proto.FileUploader_pb2 import FileUploader
from streamlit.proto.MultiSelect_pb2 import MultiSelect
from streamlit.proto.NumberInput_pb2 import NumberInput
from streamlit.proto.Radio_pb2 import Radio
from streamlit.proto.Selectbox_pb2 import Selectbox
from streamlit.proto.Slider_pb2 import Slider
from streamlit.proto.TextArea_pb2 import TextArea
from streamlit.proto.TextInput_pb2 import TextInput
from streamlit.proto.TimeInput_pb2 import TimeInput
from streamlit.type_util import ValueFieldName
from streamlit.util import HASHLIB_KWARGS
def compute_widget_id(element_type: str, user_key: str | None=None, **kwargs: SAFE_VALUES | Sequence[SAFE_VALUES]) -> str:
    """Compute the widget id for the given widget. This id is stable: a given
    set of inputs to this function will always produce the same widget id output.

    Only stable, deterministic values should be used to compute widget ids. Using
    nondeterministic values as inputs can cause the resulting widget id to
    change between runs.

    The widget id includes the user_key so widgets with identical arguments can
    use it to be distinct.

    The widget id includes an easily identified prefix, and the user_key as a
    suffix, to make it easy to identify it and know if a key maps to it.
    """
    h = hashlib.new('md5', **HASHLIB_KWARGS)
    h.update(element_type.encode('utf-8'))
    for k, v in kwargs.items():
        h.update(str(k).encode('utf-8'))
        h.update(str(v).encode('utf-8'))
    return f'{GENERATED_WIDGET_ID_PREFIX}-{h.hexdigest()}-{user_key}'