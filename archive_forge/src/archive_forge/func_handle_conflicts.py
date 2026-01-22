import datetime
import functools
import pytz
from oslo_db import exception as db_exception
from oslo_db import options as db_options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import models
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from osprofiler import opts as profiler
import osprofiler.sqlalchemy
import sqlalchemy as sql
from sqlalchemy.ext import declarative
from sqlalchemy.orm.attributes import flag_modified, InstrumentedAttribute
from sqlalchemy import types as sql_types
from keystone.common import driver_hints
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def handle_conflicts(conflict_type='object'):
    """Convert select sqlalchemy exceptions into HTTP 409 Conflict."""
    _conflict_msg = 'Conflict %(conflict_type)s: %(details)s'

    def decorator(method):

        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except db_exception.DBDuplicateEntry as e:
                LOG.debug(_conflict_msg, {'conflict_type': conflict_type, 'details': e})
                name = None
                field = None
                domain_id = None
                params = args[1:]
                for arg in params:
                    if isinstance(arg, dict):
                        if 'name' in arg:
                            field = 'name'
                            name = arg['name']
                        elif 'id' in arg:
                            field = 'ID'
                            name = arg['id']
                        if 'domain_id' in arg:
                            domain_id = arg['domain_id']
                msg = _('Duplicate entry')
                if name and domain_id:
                    msg = _('Duplicate entry found with %(field)s %(name)s at domain ID %(domain_id)s') % {'field': field, 'name': name, 'domain_id': domain_id}
                elif name:
                    msg = _('Duplicate entry found with %(field)s %(name)s') % {'field': field, 'name': name}
                elif domain_id:
                    msg = _('Duplicate entry at domain ID %s') % domain_id
                raise exception.Conflict(type=conflict_type, details=msg)
            except db_exception.DBError as e:
                if isinstance(e.inner_exception, IntegrityError):
                    LOG.debug(_conflict_msg, {'conflict_type': conflict_type, 'details': e})
                    raise exception.UnexpectedError(_('An unexpected error occurred when trying to store %s') % conflict_type)
                raise
        return wrapper
    return decorator