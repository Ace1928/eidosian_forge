import collections
from oslo_config import cfg
from oslo_log import log as logging
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
import tenacity
from heat.common import crypt
from heat.common import exception
from heat.common.i18n import _
from heat.db import api as db_api
from heat.objects import base as heat_base
from heat.objects import fields as heat_fields
from heat.objects import resource_data
from heat.objects import resource_properties_data as rpd
@classmethod
def get_by_name_and_stack(cls, context, resource_name, stack_id):
    resource_db = db_api.resource_get_by_name_and_stack(context, resource_name, stack_id)
    return cls._from_db_object(cls(context), context, resource_db)