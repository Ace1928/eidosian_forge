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
def image_location_delete(context, image_id, location_id, status, delete_time=None):
    if status not in ('deleted', 'pending_delete'):
        msg = _("The status of deleted image location can only be set to 'pending_delete' or 'deleted'.")
        raise exception.Invalid(msg)
    deleted = False
    for loc in DATA['locations']:
        if loc['id'] == location_id and loc['image_id'] == image_id:
            deleted = True
            delete_time = delete_time or timeutils.utcnow()
            loc.update({'deleted': deleted, 'status': status, 'updated_at': delete_time, 'deleted_at': delete_time})
            break
    if not deleted:
        msg = _('No location found with ID %(loc)s from image %(img)s') % dict(loc=location_id, img=image_id)
        LOG.warning(msg)
        raise exception.NotFound(msg)