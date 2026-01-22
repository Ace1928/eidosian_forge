from __future__ import annotations
from typing import Callable, Final, List, cast
from streamlit.logger import get_logger
from streamlit.runtime.app_session import AppSession
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.session_manager import (
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
from streamlit.watcher import LocalSourcesWatcher
def is_active_session(self, session_id: str) -> bool:
    return session_id in self._active_session_info_by_id