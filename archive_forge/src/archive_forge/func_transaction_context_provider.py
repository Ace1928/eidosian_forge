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
def transaction_context_provider(klass):
    """Decorate a class with ``session`` and ``connection`` attributes."""
    setattr(klass, 'transaction_ctx', property(_transaction_ctx_for_context))
    for attr in ('session', 'connection', 'transaction'):
        setattr(klass, attr, _context_descriptor(attr))
    return klass