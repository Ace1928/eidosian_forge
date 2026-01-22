from __future__ import annotations
import collections
import logging
from . import config
from . import engines
from . import util
from .. import exc
from .. import inspect
from ..engine import url as sa_url
from ..sql import ddl
from ..sql import schema
def for_db(self, *dbnames):

    def decorate(fn):
        if self.decorator:
            fn = self.decorator(fn)
        for dbname in dbnames:
            self.fns[dbname] = fn
        return self
    return decorate