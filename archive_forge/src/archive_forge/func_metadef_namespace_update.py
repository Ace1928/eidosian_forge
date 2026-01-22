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
def metadef_namespace_update(context, namespace_id, values):
    """Update a namespace object"""
    global DATA
    namespace_values = copy.deepcopy(values)
    namespace = metadef_namespace_get_by_id(context, namespace_id)
    if namespace['namespace'] != values['namespace']:
        for db_namespace in DATA['metadef_namespaces']:
            if db_namespace['namespace'] == values['namespace']:
                LOG.debug('Invalid update. It would result in a duplicate metadata definition namespace with the same name of %s', values['namespace'])
                emsg = _('Invalid update. It would result in a duplicate metadata definition namespace with the same name of %s') % values['namespace']
                raise exception.MetadefDuplicateNamespace(emsg)
    DATA['metadef_namespaces'].remove(namespace)
    namespace.update(namespace_values)
    namespace['updated_at'] = timeutils.utcnow()
    DATA['metadef_namespaces'].append(namespace)
    return namespace