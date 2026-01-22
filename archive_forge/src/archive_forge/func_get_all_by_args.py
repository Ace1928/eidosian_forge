from oslo_versionedobjects import base
from oslo_versionedobjects import fields
from heat.common import service_utils
from heat.db import api as db_api
from heat.objects import base as heat_base
@classmethod
def get_all_by_args(cls, context, host, binary, hostname):
    return cls._from_db_objects(context, db_api.service_get_all_by_args(context, host, binary, hostname))