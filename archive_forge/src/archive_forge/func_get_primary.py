from __future__ import annotations
import os
import queue
import random
import time
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, cast
from pymongo import _csot, common, helpers, periodic_executor
from pymongo.client_session import _ServerSession, _ServerSessionPool
from pymongo.errors import (
from pymongo.hello import Hello
from pymongo.lock import _create_lock
from pymongo.monitor import SrvMonitor
from pymongo.pool import Pool, PoolOptions
from pymongo.server import Server
from pymongo.server_description import ServerDescription
from pymongo.server_selectors import (
from pymongo.topology_description import (
def get_primary(self) -> Optional[_Address]:
    """Return primary's address or None."""
    with self._lock:
        topology_type = self._description.topology_type
        if topology_type != TOPOLOGY_TYPE.ReplicaSetWithPrimary:
            return None
        return writable_server_selector(self._new_selection())[0].address