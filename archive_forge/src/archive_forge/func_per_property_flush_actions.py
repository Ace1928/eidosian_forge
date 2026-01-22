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
def per_property_flush_actions(self, uow):
    parent_saves = unitofwork.SaveUpdateAll(uow, self.parent.base_mapper)
    after_save = unitofwork.ProcessAll(uow, self, False, False)
    uow.dependencies.update([(parent_saves, after_save)])