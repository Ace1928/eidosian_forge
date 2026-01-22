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
def per_property_preprocessors(self, uow):
    if self.prop._reverse_property:
        if self.passive_updates:
            return
        elif False in (prop.passive_updates for prop in self.prop._reverse_property):
            return
    uow.register_preprocessor(self, False)