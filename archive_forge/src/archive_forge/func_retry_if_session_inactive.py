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
def retry_if_session_inactive(context_var_name='context'):
    """Retries only if the session in the context is inactive.

    Calls a retry_db_errors wrapped version of the function if the context's
    session passed in is inactive, otherwise it just calls the function
    directly. This is useful to avoid retrying things inside of a transaction
    which is ineffective for DB races/errors.

    This should be used in all cases where retries are desired and the method
    accepts a context.
    """

    def decorator(f):
        try:
            ctx_arg_index = p_util.getargspec(f).args.index(context_var_name)
        except ValueError as e:
            msg = _('Could not find position of var %s') % context_var_name
            raise RuntimeError(msg) from e
        f_with_retry = retry_db_errors(f)

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            context = kwargs.get(context_var_name)
            if context is None:
                context = args[ctx_arg_index]
            if context.session and is_session_active(context.session):
                return f(*args, **kwargs)
            else:
                return f_with_retry(*args, **kwargs, _context_reference=context)
        return wrapped
    return decorator