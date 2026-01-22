import datetime
import itertools
import threading
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import session as oslo_db_session
from oslo_log import log as logging
from oslo_utils import excutils
import osprofiler.sqlalchemy
from retrying import retry
import sqlalchemy
from sqlalchemy.ext.compiler import compiles
from sqlalchemy import MetaData, Table
import sqlalchemy.orm as sa_orm
from sqlalchemy import sql
import sqlalchemy.sql as sa_sql
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.db.sqlalchemy.metadef_api import (resource_type
from glance.db.sqlalchemy.metadef_api import (resource_type_association
from glance.db.sqlalchemy.metadef_api import namespace as metadef_namespace_api
from glance.db.sqlalchemy.metadef_api import object as metadef_object_api
from glance.db.sqlalchemy.metadef_api import property as metadef_property_api
from glance.db.sqlalchemy.metadef_api import tag as metadef_tag_api
from glance.db.sqlalchemy import models
from glance.db import utils as db_utils
from glance.i18n import _, _LW, _LI, _LE
def purge_deleted_rows(context, age_in_days, max_rows):
    """Purges soft deleted rows

    Deletes rows of table images, table tasks and all dependent tables
    according to given age for relevant models.
    """
    _validate_db_int(max_rows=max_rows)
    session = get_session()
    metadata = MetaData()
    engine = get_engine()
    deleted_age = timeutils.utcnow() - datetime.timedelta(days=age_in_days)
    tables = []
    for model_class in models.__dict__.values():
        if not hasattr(model_class, '__tablename__'):
            continue
        if hasattr(model_class, 'deleted'):
            tables.append(model_class.__tablename__)
    t = Table('tasks', metadata, autoload_with=engine)
    ti = Table('task_info', metadata, autoload_with=engine)
    joined_rec = ti.join(t, t.c.id == ti.c.task_id)
    deleted_task_info = sql.select(ti.c.task_id).where(t.c.deleted_at < deleted_age).select_from(joined_rec).order_by(t.c.deleted_at)
    if max_rows != -1:
        deleted_task_info = deleted_task_info.limit(max_rows)
    delete_statement = DeleteFromSelect(ti, deleted_task_info, ti.c.task_id)
    LOG.info(_LI('Purging deleted rows older than %(age_in_days)d day(s) from table %(tbl)s'), {'age_in_days': age_in_days, 'tbl': ti})
    try:
        with session.begin():
            result = session.execute(delete_statement)
    except (db_exception.DBError, db_exception.DBReferenceError) as ex:
        LOG.exception(_LE('DBError detected when force purging table=%(table)s: %(error)s'), {'table': ti, 'error': str(ex)})
        raise
    rows = result.rowcount
    LOG.info(_LI('Deleted %(rows)d row(s) from table %(tbl)s'), {'rows': rows, 'tbl': ti})
    for tbl in ('images', 'tasks'):
        try:
            tables.remove(tbl)
        except ValueError:
            LOG.warning(_LW('Expected table %(tbl)s was not found in DB.'), {'tbl': tbl})
        else:
            if tbl == 'images':
                continue
            tables.append(tbl)
    for tbl in tables:
        tab = Table(tbl, metadata, autoload_with=engine)
        LOG.info(_LI('Purging deleted rows older than %(age_in_days)d day(s) from table %(tbl)s'), {'age_in_days': age_in_days, 'tbl': tbl})
        column = tab.c.id
        deleted_at_column = tab.c.deleted_at
        query_delete = sql.select(column).where(deleted_at_column < deleted_age).order_by(deleted_at_column)
        if max_rows != -1:
            query_delete = query_delete.limit(max_rows)
        delete_statement = DeleteFromSelect(tab, query_delete, column)
        try:
            with session.begin():
                result = session.execute(delete_statement)
        except db_exception.DBReferenceError as ex:
            with excutils.save_and_reraise_exception():
                LOG.error(_LE('DBError detected when purging from %(tablename)s: %(error)s'), {'tablename': tbl, 'error': str(ex)})
        rows = result.rowcount
        LOG.info(_LI('Deleted %(rows)d row(s) from table %(tbl)s'), {'rows': rows, 'tbl': tbl})