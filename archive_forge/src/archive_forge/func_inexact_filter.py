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
def inexact_filter(model, query, filter_, satisfied_filters):
    """Apply an inexact filter to a query.

        :param model: the table model in question
        :param query: query to apply filters to
        :param dict filter_: describes this filter
        :param list satisfied_filters: filter_ will be added if it is
                                       satisfied.

        :returns: query updated to add any inexact filters satisfied

        """
    column_attr = getattr(model, filter_['name'])
    if filter_['case_sensitive']:
        return query
    if filter_['comparator'] == 'contains':
        _WontMatch.check(filter_['value'], column_attr)
        query_term = column_attr.ilike('%%%s%%' % filter_['value'])
    elif filter_['comparator'] == 'startswith':
        _WontMatch.check(filter_['value'], column_attr)
        query_term = column_attr.ilike('%s%%' % filter_['value'])
    elif filter_['comparator'] == 'endswith':
        _WontMatch.check(filter_['value'], column_attr)
        query_term = column_attr.ilike('%%%s' % filter_['value'])
    else:
        return query
    satisfied_filters.append(filter_)
    return query.filter(query_term)