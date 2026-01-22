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
def metadef_tag_create_tags(context, namespace_name, tag_list, can_append=False):
    """Create a metadef tag"""
    global DATA
    namespace = metadef_namespace_get(context, namespace_name)
    _check_namespace_visibility(context, namespace, namespace_name)
    required_attributes = ['name']
    allowed_attributes = ['name']
    data_tag_list = []
    tag_name_list = []
    if can_append:
        tag_name_list = [tag['name'] for tag in metadef_tag_get_all(context, namespace_name)]
    for tag_value in tag_list:
        tag_values = copy.deepcopy(tag_value)
        tag_name = tag_values['name']
        for key in required_attributes:
            if key not in tag_values:
                raise exception.Invalid('%s is a required attribute' % key)
        incorrect_keys = set(tag_values.keys()) - set(allowed_attributes)
        if incorrect_keys:
            raise exception.Invalid('The keys %s are not valid' % str(incorrect_keys))
        if tag_name in tag_name_list:
            LOG.debug('A metadata definition tag with name=%(name)s in namespace=%(namespace_name)s already exists.', {'name': tag_name, 'namespace_name': namespace_name})
            raise exception.MetadefDuplicateTag(name=tag_name, namespace_name=namespace_name)
        else:
            tag_name_list.append(tag_name)
        tag_values['namespace_id'] = namespace['id']
        data_tag_list.append(_format_tag(tag_values))
    if not can_append:
        DATA['metadef_tags'] = []
    for tag in data_tag_list:
        DATA['metadef_tags'].append(tag)
    return data_tag_list