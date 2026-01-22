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
def metadef_resource_type_create(context, values):
    """Create a metadef resource type"""
    global DATA
    resource_type_values = copy.deepcopy(values)
    resource_type_name = resource_type_values['name']
    allowed_attrubites = ['name', 'protected']
    for resource_type in DATA['metadef_resource_types']:
        if resource_type['name'] == resource_type_name:
            raise exception.Duplicate()
    incorrect_keys = set(resource_type_values.keys()) - set(allowed_attrubites)
    if incorrect_keys:
        raise exception.Invalid('The keys %s are not valid' % str(incorrect_keys))
    resource_type = _format_resource_type(resource_type_values)
    DATA['metadef_resource_types'].append(resource_type)
    return resource_type