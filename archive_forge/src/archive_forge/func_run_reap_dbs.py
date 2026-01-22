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
def run_reap_dbs(url, ident):
    """Remove databases that were created during the test process, after the
    process has ended.

    This is an optional step that is invoked for certain backends that do not
    reliably release locks on the database as long as a process is still in
    use. For the internal dialects, this is currently only necessary for
    mssql and oracle.
    """