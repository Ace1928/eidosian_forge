from oslo_versionedobjects import base
from oslo_versionedobjects import fields
from heat.db import api as db_api
from heat.objects import base as heat_base
from heat.objects import fields as heat_fields
@classmethod
def get_snapshot_by_stack(cls, context, snapshot_id, stack):
    return cls._from_db_object(context, cls(), db_api.snapshot_get_by_stack(context, snapshot_id, stack))