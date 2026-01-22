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
def metadef_object_delete_namespace_content(context, namespace_name):
    """Delete an object or raise if namespace or object doesn't exist."""
    return _metadef_delete_namespace_content(metadef_object_get_all, 'metadef_objects', context, namespace_name)