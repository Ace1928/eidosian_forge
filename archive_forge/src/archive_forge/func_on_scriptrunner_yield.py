from __future__ import annotations
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import cast
from streamlit import util
from streamlit.proto.WidgetStates_pb2 import WidgetStates
from streamlit.runtime.state import coalesce_widget_states
def on_scriptrunner_yield(self) -> ScriptRequest | None:
    """Called by the ScriptRunner when it's at a yield point.

        If we have no request or a RERUN request corresponding to one or more fragments,
        return None.

        If we have a (full script) RERUN request, return the request and set our internal
        state to CONTINUE.

        If we have a STOP request, return the request and remain stopped.
        """
    if self._state == ScriptRequestType.CONTINUE or (self._state == ScriptRequestType.RERUN and self._rerun_data.fragment_id_queue):
        return None
    with self._lock:
        if self._state == ScriptRequestType.RERUN:
            if self._rerun_data.fragment_id_queue:
                return None
            self._state = ScriptRequestType.CONTINUE
            return ScriptRequest(ScriptRequestType.RERUN, self._rerun_data)
        assert self._state == ScriptRequestType.STOP
        return ScriptRequest(ScriptRequestType.STOP)