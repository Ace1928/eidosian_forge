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
@utils.no_4byte_params
def metadef_namespace_create(context, values):
    """Create a namespace object"""
    global DATA
    namespace_values = copy.deepcopy(values)
    namespace_name = namespace_values.get('namespace')
    required_attributes = ['namespace', 'owner']
    allowed_attributes = ['namespace', 'owner', 'display_name', 'description', 'visibility', 'protected']
    for namespace in DATA['metadef_namespaces']:
        if namespace['namespace'] == namespace_name:
            LOG.debug('Can not create the metadata definition namespace. Namespace=%s already exists.', namespace_name)
            raise exception.MetadefDuplicateNamespace(namespace_name=namespace_name)
    for key in required_attributes:
        if key not in namespace_values:
            raise exception.Invalid('%s is a required attribute' % key)
    incorrect_keys = set(namespace_values.keys()) - set(allowed_attributes)
    if incorrect_keys:
        raise exception.Invalid('The keys %s are not valid' % str(incorrect_keys))
    namespace = _format_namespace(namespace_values)
    DATA['metadef_namespaces'].append(namespace)
    return namespace