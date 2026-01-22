from __future__ import annotations
import hashlib
from typing import TYPE_CHECKING, Final, MutableMapping
from weakref import WeakKeyDictionary
from streamlit import config, util
from streamlit.logger import get_logger
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.util import HASHLIB_KWARGS
def remove_expired_entries_for_session(self, session: AppSession, script_run_count: int) -> None:
    """Remove any cached messages that have expired from the given session.

        This should be called each time a AppSession finishes executing.

        Parameters
        ----------
        session : AppSession
        script_run_count : int
            The number of times the session's script has run

        """
    max_age = config.get_option('global.maxCachedMessageAge')
    for msg_hash, entry in self._entries.copy().items():
        if not entry.has_session_ref(session):
            continue
        age = entry.get_session_ref_age(session, script_run_count)
        if age > max_age:
            _LOGGER.debug('Removing expired entry [session=%s, hash=%s, age=%s]', id(session), msg_hash, age)
            entry.remove_session_ref(session)
            if not entry.has_refs():
                del self._entries[msg_hash]