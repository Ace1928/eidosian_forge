import contextlib
import functools
import inspect
import operator
import threading
import warnings
import debtcollector.moves
import debtcollector.removals
import debtcollector.renames
from oslo_config import cfg
from oslo_utils import excutils
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import orm
from oslo_db import warning
@property
def async_(self):
    """Modifier to set a READER operation to ASYNC_READER."""
    if self._mode is _WRITER:
        raise TypeError('Setting async on a WRITER makes no sense')
    return self._clone(mode=_ASYNC_READER)