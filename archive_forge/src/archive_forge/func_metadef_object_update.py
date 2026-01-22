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
def metadef_object_update(context, namespace_name, object_id, values):
    """Update a metadef object"""
    global DATA
    namespace = metadef_namespace_get(context, namespace_name)
    _check_namespace_visibility(context, namespace, namespace_name)
    object = metadef_object_get_by_id(context, namespace_name, object_id)
    if object['name'] != values['name']:
        for db_object in DATA['metadef_objects']:
            if db_object['name'] == values['name'] and db_object['namespace_id'] == namespace['id']:
                LOG.debug('Invalid update. It would result in a duplicate metadata definition object with same name=%(name)s in namespace=%(namespace_name)s.', {'name': object['name'], 'namespace_name': namespace_name})
                emsg = _('Invalid update. It would result in a duplicate metadata definition object with the same name=%(name)s  in namespace=%(namespace_name)s.') % {'name': object['name'], 'namespace_name': namespace_name}
                raise exception.MetadefDuplicateObject(emsg)
    DATA['metadef_objects'].remove(object)
    object.update(values)
    object['updated_at'] = timeutils.utcnow()
    DATA['metadef_objects'].append(object)
    return object