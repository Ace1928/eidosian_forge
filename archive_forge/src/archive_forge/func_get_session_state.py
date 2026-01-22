from __future__ import annotations
from typing import Any, Final, Iterator, MutableMapping
from streamlit import logger as _logger
from streamlit import runtime
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.state.common import require_valid_user_key
from streamlit.runtime.state.safe_session_state import SafeSessionState
from streamlit.runtime.state.session_state import SessionState
from streamlit.type_util import Key
def get_session_state() -> SafeSessionState:
    """Get the SessionState object for the current session.

    Note that in streamlit scripts, this function should not be called
    directly. Instead, SessionState objects should be accessed via
    st.session_state.
    """
    global _state_use_warning_already_displayed
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    ctx = get_script_run_ctx()
    if ctx is None:
        if not _state_use_warning_already_displayed:
            _state_use_warning_already_displayed = True
            if not runtime.exists():
                _LOGGER.warning('Session state does not function when running a script without `streamlit run`')
        return SafeSessionState(SessionState(), lambda: None)
    return ctx.session_state