from __future__ import annotations
import os
import time
import types
from typing import Any
from urllib import parse
from streamlit import runtime
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.WidgetStates_pb2 import WidgetStates
from streamlit.runtime.forward_msg_queue import ForwardMsgQueue
from streamlit.runtime.fragment import MemoryFragmentStorage
from streamlit.runtime.memory_uploaded_file_manager import MemoryUploadedFileManager
from streamlit.runtime.scriptrunner import RerunData, ScriptRunner, ScriptRunnerEvent
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.scriptrunner.script_run_context import ScriptRunContext
from streamlit.runtime.state.safe_session_state import SafeSessionState
from streamlit.testing.v1.element_tree import ElementTree, parse_tree_from_messages
Run the script, and parse the output messages for querying
        and interaction.

        Timeout is in seconds.
        