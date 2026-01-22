from oslo_db import exception as db_exc
from oslo_log import log as logging
from sqlalchemy import func
import sqlalchemy.orm as sa_orm
from glance.common import exception as exc
from glance.db.sqlalchemy.metadef_api import namespace as namespace_api
import glance.db.sqlalchemy.metadef_api.utils as metadef_utils
from glance.db.sqlalchemy import models_metadef as models
from glance.i18n import _
Get the count of objects for a namespace, raise if ns not found