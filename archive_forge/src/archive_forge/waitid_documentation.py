import errno
import math
import os
import sys
from typing import TYPE_CHECKING
from .. import _core, _subprocess
from .._sync import CapacityLimiter, Event
from .._threads import to_thread_run_sync
Spawn a thread that waits for ``pid`` to exit, then wake any tasks
    that were waiting on it.
    