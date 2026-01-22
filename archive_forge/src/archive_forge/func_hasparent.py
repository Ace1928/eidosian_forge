from __future__ import annotations
from . import attributes
from . import exc
from . import sync
from . import unitofwork
from . import util as mapperutil
from .interfaces import MANYTOMANY
from .interfaces import MANYTOONE
from .interfaces import ONETOMANY
from .. import exc as sa_exc
from .. import sql
from .. import util
def hasparent(self, state):
    """return True if the given object instance has a parent,
        according to the ``InstrumentedAttribute`` handled by this
        ``DependencyProcessor``.

        """
    return self.parent.class_manager.get_impl(self.key).hasparent(state)