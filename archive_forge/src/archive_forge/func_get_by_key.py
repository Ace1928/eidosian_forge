from oslo_versionedobjects import base
from oslo_versionedobjects import fields
from heat.db import api as db_api
from heat.objects import base as heat_base
from heat.objects import fields as heat_fields
@classmethod
def get_by_key(cls, context, entity_id, traversal_id, is_update):
    sync_point_db = db_api.sync_point_get(context, entity_id, traversal_id, is_update)
    return cls._from_db_object(context, cls(), sync_point_db)