import copy
import functools
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.db import utils as db_utils
from glance.i18n import _, _LI, _LW
@log_call
def metadef_resource_type_association_delete(context, namespace_name, resource_type_name):
    global DATA
    resource_type = metadef_resource_type_association_get(context, namespace_name, resource_type_name)
    DATA['metadef_namespace_resource_types'].remove(resource_type)
    return resource_type