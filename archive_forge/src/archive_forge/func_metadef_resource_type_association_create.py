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
def metadef_resource_type_association_create(context, namespace_name, values):
    global DATA
    association_values = copy.deepcopy(values)
    namespace = metadef_namespace_get(context, namespace_name)
    resource_type_name = association_values['name']
    resource_type = metadef_resource_type_get(context, resource_type_name)
    required_attributes = ['name', 'properties_target', 'prefix']
    allowed_attributes = copy.deepcopy(required_attributes)
    for association in DATA['metadef_namespace_resource_types']:
        if association['namespace_id'] == namespace['id'] and association['resource_type'] == resource_type['id']:
            LOG.debug('The metadata definition resource-type association of resource_type=%(resource_type_name)s to namespace=%(namespace_name)s, already exists.', {'resource_type_name': resource_type_name, 'namespace_name': namespace_name})
            raise exception.MetadefDuplicateResourceTypeAssociation(resource_type_name=resource_type_name, namespace_name=namespace_name)
    for key in required_attributes:
        if key not in association_values:
            raise exception.Invalid('%s is a required attribute' % key)
    incorrect_keys = set(association_values.keys()) - set(allowed_attributes)
    if incorrect_keys:
        raise exception.Invalid('The keys %s are not valid' % str(incorrect_keys))
    association = _format_association(namespace, resource_type, association_values)
    DATA['metadef_namespace_resource_types'].append(association)
    return association