from __future__ import annotations
import hashlib
from typing import TYPE_CHECKING, Final, MutableMapping
from weakref import WeakKeyDictionary
from streamlit import config, util
from streamlit.logger import get_logger
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.util import HASHLIB_KWARGS
def remove_refs_for_session(self, session: AppSession) -> None:
    """Remove refs for all entries for the given session.

        This should be called when an AppSession is disconnected or closed.

        Parameters
        ----------
        session : AppSession
        """
    for msg_hash, entry in self._entries.copy().items():
        if entry.has_session_ref(session):
            entry.remove_session_ref(session)
        if not entry.has_refs():
            del self._entries[msg_hash]