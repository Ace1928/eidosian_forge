from oslo_log import log as logging
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.db import api as db_api
from heat.objects import base as heat_base
from heat.objects import fields as heat_fields
from heat.objects import raw_template
from heat.objects import stack_tag
@classmethod
def count_all(cls, context, **kwargs):
    return db_api.stack_count_all(context, **kwargs)