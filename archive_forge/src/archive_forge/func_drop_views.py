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
def drop_views(cfg, eng):
    inspector = inspect(eng)
    try:
        view_names = inspector.get_view_names()
    except NotImplementedError:
        pass
    else:
        with eng.begin() as conn:
            for vname in view_names:
                conn.execute(ddl._DropView(schema.Table(vname, schema.MetaData())))
    if config.requirements.schemas.enabled_for_config(cfg):
        try:
            view_names = inspector.get_view_names(schema=cfg.test_schema)
        except NotImplementedError:
            pass
        else:
            with eng.begin() as conn:
                for vname in view_names:
                    conn.execute(ddl._DropView(schema.Table(vname, schema.MetaData(), schema=cfg.test_schema)))