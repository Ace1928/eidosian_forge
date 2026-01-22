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
def metadef_property_get_all(context, namespace_name):
    """Get a metadef properties list"""
    namespace = metadef_namespace_get(context, namespace_name)
    properties = []
    _check_namespace_visibility(context, namespace, namespace_name)
    for property in DATA['metadef_properties']:
        if property['namespace_id'] == namespace['id']:
            properties.append(property)
    return properties