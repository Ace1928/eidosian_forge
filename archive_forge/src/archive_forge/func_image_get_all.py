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
def image_get_all(context, filters=None, marker=None, limit=None, sort_key=None, sort_dir=None, member_status='accepted', is_public=None, admin_as_user=False, return_tag=False, v1_mode=False):
    filters = filters or {}
    images = DATA['images'].values()
    images = _filter_images(images, filters, context, member_status, is_public, admin_as_user)
    images = _sort_images(images, sort_key, sort_dir)
    images = _do_pagination(context, images, marker, limit, filters.get('deleted'))
    force_show_deleted = True if filters.get('deleted') else False
    res = []
    for image in images:
        img = _normalize_locations(context, copy.deepcopy(image), force_show_deleted=force_show_deleted)
        if return_tag:
            img['tags'] = image_tag_get_all(context, img['id'])
        if v1_mode:
            img = db_utils.mutate_image_dict_to_v1(img)
        res.append(img)
    return res