from __future__ import annotations
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING, Any, Hashable, Iterable, Sequence, cast, overload
import streamlit
from streamlit import config, runtime, type_util
from streamlit.elements.form import is_in_form
from streamlit.errors import StreamlitAPIException
from streamlit.proto.LabelVisibilityMessage_pb2 import LabelVisibilityMessage
from streamlit.runtime.state import WidgetCallback, get_session_state
from streamlit.runtime.state.common import RegisterWidgetResult
from streamlit.type_util import T
def maybe_coerce_enum(register_widget_result, options, opt_sequence):
    """Maybe Coerce a RegisterWidgetResult with an Enum member value to
    RegisterWidgetResult[option] if option is an EnumType, otherwise just return
    the original RegisterWidgetResult."""
    if not isinstance(register_widget_result.value, Enum):
        return register_widget_result
    coerce_class: EnumMeta | None
    if isinstance(options, EnumMeta):
        coerce_class = options
    else:
        coerce_class = _extract_common_class_from_iter(opt_sequence)
        if coerce_class is None:
            return register_widget_result
    return RegisterWidgetResult(type_util.coerce_enum(register_widget_result.value, coerce_class), register_widget_result.value_changed)