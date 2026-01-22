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
def post_configure_engine(url, engine, follower_ident):
    """Perform extra steps after configuring an engine for testing.

    (For the internal dialects, currently only used by sqlite, oracle)
    """