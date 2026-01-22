import re
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import units
import glance.async_
from glance.common import exception
from glance.i18n import _, _LE, _LW
@lockutils.synchronized(lock_name)
def memoizer(lock_name):
    if lock_name not in _CACHED_THREAD_POOL:
        _CACHED_THREAD_POOL[lock_name] = func()
    return _CACHED_THREAD_POOL[lock_name]