from oslo_log import log as logging
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
from heat.common import identifier
from heat.db import api as db_api
from heat.objects import base as heat_base
from heat.objects import resource_properties_data as rpd
@classmethod
def count_all_by_stack(cls, context, stack_id):
    return db_api.event_count_all_by_stack(context, stack_id)