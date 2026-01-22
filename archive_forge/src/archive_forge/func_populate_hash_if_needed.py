from __future__ import annotations
import hashlib
from typing import TYPE_CHECKING, Final, MutableMapping
from weakref import WeakKeyDictionary
from streamlit import config, util
from streamlit.logger import get_logger
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.stats import CacheStat, CacheStatsProvider, group_stats
from streamlit.util import HASHLIB_KWARGS
def populate_hash_if_needed(msg: ForwardMsg) -> str:
    """Computes and assigns the unique hash for a ForwardMsg.

    If the ForwardMsg already has a hash, this is a no-op.

    Parameters
    ----------
    msg : ForwardMsg

    Returns
    -------
    string
        The message's hash, returned here for convenience. (The hash
        will also be assigned to the ForwardMsg; callers do not need
        to do this.)

    """
    if msg.hash == '':
        metadata = msg.metadata
        msg.ClearField('metadata')
        hasher = hashlib.md5(**HASHLIB_KWARGS)
        hasher.update(msg.SerializeToString())
        msg.hash = hasher.hexdigest()
        msg.metadata.CopyFrom(metadata)
    return msg.hash