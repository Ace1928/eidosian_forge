from oslo_db import exception as db_exc
from oslo_log import log as logging
import sqlalchemy.orm as sa_orm
from glance.common import exception as exc
from glance.db.sqlalchemy.metadef_api import namespace as namespace_api
from glance.db.sqlalchemy.metadef_api import resource_type as resource_type_api
from glance.db.sqlalchemy.metadef_api import utils as metadef_utils
from glance.db.sqlalchemy import models_metadef as models
def delete_namespace_content(context, session, namespace_id):
    """Use this def only if the ns for the id has been verified as visible"""
    count = 0
    query = session.query(models.MetadefNamespaceResourceType).filter_by(namespace_id=namespace_id)
    count = query.delete(synchronize_session='fetch')
    return count