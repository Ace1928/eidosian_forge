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
def metadef_tag_get_all(context, namespace_name, filters=None, marker=None, limit=None, sort_key='created_at', sort_dir=None, session=None):
    """Get a metadef tags list"""
    namespace = metadef_namespace_get(context, namespace_name)
    _check_namespace_visibility(context, namespace, namespace_name)
    tags = []
    for tag in DATA['metadef_tags']:
        if tag['namespace_id'] == namespace['id']:
            tags.append(tag)
    return tags