from oslo_db import exception as db_exc
from oslo_log import log as logging
import sqlalchemy.orm as sa_orm
from glance.common import exception as exc
from glance.db.sqlalchemy.metadef_api import namespace as namespace_api
from glance.db.sqlalchemy.metadef_api import resource_type as resource_type_api
from glance.db.sqlalchemy.metadef_api import utils as metadef_utils
from glance.db.sqlalchemy import models_metadef as models
def get_all_by_namespace(context, session, namespace_name):
    """List resource_type associations by namespace, raise if not found"""
    namespace = namespace_api.get(context, session, namespace_name)
    db_recs = session.query(models.MetadefResourceType).join(models.MetadefResourceType.associations).filter_by(namespace_id=namespace['id']).with_entities(models.MetadefResourceType.name, models.MetadefNamespaceResourceType.properties_target, models.MetadefNamespaceResourceType.prefix, models.MetadefNamespaceResourceType.created_at, models.MetadefNamespaceResourceType.updated_at)
    model_dict_list = []
    for name, properties_target, prefix, created_at, updated_at in db_recs:
        model_dict_list.append(_set_model_dict(name, properties_target, prefix, created_at, updated_at))
    return model_dict_list