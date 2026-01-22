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
def user_get_storage_usage(context, owner_id, image_id=None, session=None):
    images = image_get_all(context, filters={'owner': owner_id})
    total = 0
    for image in images:
        if image['status'] in ['killed', 'deleted']:
            continue
        if image['id'] != image_id:
            locations = [loc for loc in image['locations'] if loc.get('status') != 'deleted']
            total += image['size'] * len(locations)
    return total