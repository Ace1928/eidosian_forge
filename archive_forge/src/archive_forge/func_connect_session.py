from __future__ import annotations
from typing import Callable, Final, List, cast
from streamlit.logger import get_logger
from streamlit.runtime.app_session import AppSession
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.session_manager import (
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
from streamlit.watcher import LocalSourcesWatcher
def connect_session(self, client: SessionClient, script_data: ScriptData, user_info: dict[str, str | None], existing_session_id: str | None=None, session_id_override: str | None=None) -> str:
    assert not (existing_session_id and session_id_override), 'Only one of existing_session_id and session_id_override should be truthy'
    if existing_session_id in self._active_session_info_by_id:
        _LOGGER.warning('Session with id %s is already connected! Connecting to a new session.', existing_session_id)
    session_info = existing_session_id and existing_session_id not in self._active_session_info_by_id and self._session_storage.get(existing_session_id)
    if session_info:
        existing_session = session_info.session
        existing_session.register_file_watchers()
        self._active_session_info_by_id[existing_session.id] = ActiveSessionInfo(client, existing_session, session_info.script_run_count)
        self._session_storage.delete(existing_session.id)
        return existing_session.id
    session = AppSession(script_data=script_data, uploaded_file_manager=self._uploaded_file_mgr, script_cache=self._script_cache, message_enqueued_callback=self._message_enqueued_callback, local_sources_watcher=LocalSourcesWatcher(script_data.main_script_path), user_info=user_info, session_id_override=session_id_override)
    _LOGGER.debug('Created new session for client %s. Session ID: %s', id(client), session.id)
    assert session.id not in self._active_session_info_by_id, f"session.id '{session.id}' registered multiple times!"
    self._active_session_info_by_id[session.id] = ActiveSessionInfo(client, session)
    return session.id