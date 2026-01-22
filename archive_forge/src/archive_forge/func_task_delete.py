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
def task_delete(context, task_id):
    global DATA
    try:
        DATA['tasks'][task_id]['deleted'] = True
        DATA['tasks'][task_id]['deleted_at'] = timeutils.utcnow()
        DATA['tasks'][task_id]['updated_at'] = timeutils.utcnow()
        return copy.deepcopy(DATA['tasks'][task_id])
    except KeyError:
        LOG.debug('No task found with ID %s', task_id)
        raise exception.TaskNotFound(task_id=task_id)