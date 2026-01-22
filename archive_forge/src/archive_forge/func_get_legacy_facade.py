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
def get_legacy_facade(self):
    """Return a :class:`.LegacyEngineFacade` for factory from this context.

        This facade will make use of the same engine and sessionmaker
        as this factory, however will not share the same transaction context;
        the legacy facade continues to work the old way of returning
        a new Session each time get_session() is called.
        """
    return self._factory.get_legacy_facade()