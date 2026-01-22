from __future__ import annotations
import hashlib
from typing import TYPE_CHECKING, Final, MutableMapping
from weakref import WeakKeyDictionary
from streamlit import config, util
from streamlit.logger import get_logger
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.util import HASHLIB_KWARGS
def get_session_ref_age(self, session: AppSession, script_run_count: int) -> int:
    """The age of the given session's reference to the Entry,
            given a new script_run_count.

            """
    return script_run_count - self._session_script_run_counts[session]