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
def check_session_state_rules(default_value: Any, key: str | None, writes_allowed: bool=True) -> None:
    global _shown_default_value_warning
    if key is None or not runtime.exists():
        return
    session_state = get_session_state()
    if not session_state.is_new_state_value(key):
        return
    if not writes_allowed:
        raise StreamlitAPIException(SESSION_STATE_WRITES_NOT_ALLOWED_ERROR_TEXT)
    if default_value is not None and (not _shown_default_value_warning) and (not config.get_option('global.disableWidgetStateDuplicationWarning')):
        streamlit.warning(f'The widget with key "{key}" was created with a default value but also had its value set via the Session State API.')
        _shown_default_value_warning = True