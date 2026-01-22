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
def insert_cache_details(context, node_reference_url, image_id, size, checksum=None, last_accessed=None, last_modified=None, hits=None):
    global DATA
    node_reference = node_reference_get_by_url(context, node_reference_url)
    accessed = last_accessed or timeutils.utcnow()
    modified = last_modified or timeutils.utcnow()
    values = {'last_accessed': accessed, 'last_modified': modified, 'node_reference_id': node_reference['node_reference_id'], 'checksum': checksum, 'image_id': image_id, 'size': size, 'hits': hits or 0, 'id': str(uuid.uuid4())}
    if image_id in DATA['cached_images']:
        raise exception.Duplicate()
    DATA['cached_images'][image_id] = values