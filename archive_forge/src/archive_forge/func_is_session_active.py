import contextlib
import copy
import functools
import weakref
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_db import api as oslo_db_api
from oslo_db import exception as db_exc
from oslo_db.sqlalchemy import enginefacade
from oslo_log import log as logging
from oslo_utils import excutils
from osprofiler import opts as profiler_opts
import osprofiler.sqlalchemy
from pecan import util as p_util
import sqlalchemy
from sqlalchemy import event  # noqa
from sqlalchemy import exc as sql_exc
from sqlalchemy import orm
from sqlalchemy.orm import exc
from neutron_lib._i18n import _
from neutron_lib.db import model_base
from neutron_lib import exceptions
from neutron_lib.objects import exceptions as obj_exc
def is_session_active(session):
    """Return if the session is active

    Since sqlalchemy 1.4, "autocommit=False" by default; in sqlalchemy 2.0,
    that will be the only possible value. If session autocommit is False, the
    session transaction will not end at the end of a reader/writer context.
    In this case, a session could have an active transaction even when it is
    not inside a reader/writer context. In order to mimic the previous
    behaviour, this method checks if there is a transaction created and if
    the transaction has any active connection against the database server.
    """
    if getattr(session, 'autocommit', None):
        return session.is_active
    if not session.get_transaction():
        return False
    if not session.get_transaction()._connections:
        return False
    return True