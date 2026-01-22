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
def image_restore(context, image_id):
    """Restore the pending-delete image to active."""
    image = _image_get(context, image_id)
    if image['status'] != 'pending_delete':
        msg = _('cannot restore the image from %s to active (wanted from_state=pending_delete)') % image['status']
        raise exception.Conflict(msg)
    values = {'status': 'active', 'deleted': 0}
    image_update(context, image_id, values)