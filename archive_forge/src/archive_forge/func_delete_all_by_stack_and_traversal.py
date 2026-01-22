from oslo_versionedobjects import base
from oslo_versionedobjects import fields
from heat.db import api as db_api
from heat.objects import base as heat_base
from heat.objects import fields as heat_fields
@classmethod
def delete_all_by_stack_and_traversal(cls, context, stack_id, traversal_id):
    return db_api.sync_point_delete_all_by_stack_and_traversal(context, stack_id, traversal_id)