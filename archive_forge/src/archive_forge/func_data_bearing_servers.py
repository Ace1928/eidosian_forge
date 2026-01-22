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
def data_bearing_servers(self) -> list[ServerDescription]:
    """Return a list of all data-bearing servers.

        This includes any server that might be selected for an operation.
        """
    if self._description.topology_type == TOPOLOGY_TYPE.Single:
        return self._description.known_servers
    return self._description.readable_servers