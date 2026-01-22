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
def metadef_resource_type_association_get(context, namespace_name, resource_type_name):
    namespace = metadef_namespace_get(context, namespace_name)
    resource_type = metadef_resource_type_get(context, resource_type_name)
    for association in DATA['metadef_namespace_resource_types']:
        if association['namespace_id'] == namespace['id'] and association['resource_type'] == resource_type['id']:
            return association
    else:
        LOG.debug('No resource type association found associated with namespace %s and resource type %s', namespace_name, resource_type_name)
        raise exception.MetadefResourceTypeAssociationNotFound(resource_type_name=resource_type_name, namespace_name=namespace_name)