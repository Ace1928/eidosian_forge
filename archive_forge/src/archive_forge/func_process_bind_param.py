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
def process_bind_param(self, value, dialect):
    if value is None:
        return value
    else:
        if not isinstance(value, datetime.datetime):
            raise ValueError(_('Programming Error: value to be stored must be a datetime object.'))
        value = timeutils.normalize_time(value)
        value = value.replace(tzinfo=pytz.UTC)
        return int((value - self.epoch).total_seconds() * 1000000)