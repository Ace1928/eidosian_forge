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
def eq_props(self) -> tuple[tuple[_Address, ...], Optional[str], Optional[str], str]:
    """The properties to use for MongoClient/Topology equality checks."""
    ts = self._settings
    return (tuple(sorted(ts.seeds)), ts.replica_set_name, ts.fqdn, ts.srv_service_name)