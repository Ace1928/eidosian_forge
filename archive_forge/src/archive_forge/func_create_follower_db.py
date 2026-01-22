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
def create_follower_db(follower_ident):
    for cfg in _configs_for_db_operation():
        log.info('CREATE database %s, URI %r', follower_ident, cfg.db.url)
        create_db(cfg, cfg.db, follower_ident)