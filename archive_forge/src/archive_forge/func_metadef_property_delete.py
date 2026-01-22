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
def metadef_property_delete(context, namespace_name, property_name):
    """Delete a metadef property"""
    global DATA
    property = metadef_property_get(context, namespace_name, property_name)
    DATA['metadef_properties'].remove(property)
    return property