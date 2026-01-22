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
@register.init
def temp_table_keyword_args(cfg, eng):
    """Specify keyword arguments for creating a temporary Table.

    Dialect-specific implementations of this method will return the
    kwargs that are passed to the Table method when creating a temporary
    table for testing, e.g., in the define_temp_tables method of the
    ComponentReflectionTest class in suite/test_reflection.py
    """
    raise NotImplementedError('no temp table keyword args routine for cfg: %s' % (eng.url,))