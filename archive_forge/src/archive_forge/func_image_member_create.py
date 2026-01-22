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
@utils.no_4byte_params
def image_member_create(context, values):
    member = _image_member_format(values['image_id'], values['member'], values.get('can_share', False), values.get('status', 'pending'), values.get('deleted', False))
    global DATA
    DATA['members'].append(member)
    return copy.deepcopy(member)