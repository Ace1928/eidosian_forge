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
def image_member_count(context, image_id):
    """Return the number of image members for this image

    :param image_id: identifier of image entity
    """
    if not image_id:
        msg = _('Image id is required.')
        raise exception.Invalid(msg)
    members = DATA['members']
    return len([x for x in members if x['image_id'] == image_id])