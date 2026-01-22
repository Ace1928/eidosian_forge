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
def get_server_by_address(self, address: _Address) -> Optional[Server]:
    """Get a Server or None.

        Returns the current version of the server immediately, even if it's
        Unknown or absent from the topology. Only use this in unittests.
        In driver code, use select_server_by_address, since then you're
        assured a recent view of the server's type and wire protocol version.
        """
    return self._servers.get(address)