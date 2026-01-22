from __future__ import annotations
import asyncio
import sys
import uuid
from enum import Enum
from typing import TYPE_CHECKING, Callable, Final
import streamlit.elements.exception as exception_utils
from streamlit import config, runtime, source_util
from streamlit.case_converters import to_snake_case
from streamlit.logger import get_logger
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ClientState_pb2 import ClientState
from streamlit.proto.Common_pb2 import FileURLs, FileURLsRequest
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.GitInfo_pb2 import GitInfo
from streamlit.proto.NewSession_pb2 import (
from streamlit.proto.PagesChanged_pb2 import PagesChanged
from streamlit.runtime import caching, legacy_caching
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.runtime.fragment import FragmentStorage, MemoryFragmentStorage
from streamlit.runtime.metrics_util import Installation
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner import RerunData, ScriptRunner, ScriptRunnerEvent
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.secrets import secrets_singleton
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
from streamlit.version import STREAMLIT_VERSION_STRING
from streamlit.watcher import LocalSourcesWatcher
def register_file_watchers(self) -> None:
    """Register handlers to be called when various files are changed.

        Files that we watch include:
          * source files that already exist (for edits)
          * `.py` files in the the main script's `pages/` directory (for file additions
            and deletions)
          * project and user-level config.toml files
          * the project-level secrets.toml files

        This method is called automatically on AppSession construction, but it may be
        called again in the case when a session is disconnected and is being reconnect
        to.
        """
    if self._local_sources_watcher is None:
        self._local_sources_watcher = LocalSourcesWatcher(self._script_data.main_script_path)
    self._local_sources_watcher.register_file_change_callback(self._on_source_file_changed)
    self._stop_config_listener = config.on_config_parsed(self._on_source_file_changed, force_connect=True)
    self._stop_pages_listener = source_util.register_pages_changed_callback(self._on_pages_changed)
    secrets_singleton.file_change_listener.connect(self._on_secrets_file_changed)