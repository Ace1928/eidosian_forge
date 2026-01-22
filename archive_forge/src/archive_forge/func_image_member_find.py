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
def image_member_find(context, image_id=None, member=None, status=None, include_deleted=False):
    filters = []
    images = DATA['images']
    members = DATA['members']

    def is_visible(member):
        return member['member'] == context.owner or images[member['image_id']]['owner'] == context.owner
    if not context.is_admin:
        filters.append(is_visible)
    if image_id is not None:
        filters.append(lambda m: m['image_id'] == image_id)
    if member is not None:
        filters.append(lambda m: m['member'] == member)
    if status is not None:
        filters.append(lambda m: m['status'] == status)
    for f in filters:
        members = filter(f, members)
    return [copy.deepcopy(m) for m in members]