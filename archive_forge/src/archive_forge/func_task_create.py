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
def task_create(context, values):
    """Create a task object"""
    global DATA
    task_values = copy.deepcopy(values)
    task_id = task_values.get('id', str(uuid.uuid4()))
    required_attributes = ['type', 'status', 'input']
    allowed_attributes = ['id', 'type', 'status', 'input', 'result', 'owner', 'message', 'expires_at', 'created_at', 'updated_at', 'deleted_at', 'deleted', 'image_id', 'request_id', 'user_id']
    if task_id in DATA['tasks']:
        raise exception.Duplicate()
    for key in required_attributes:
        if key not in task_values:
            raise exception.Invalid('%s is a required attribute' % key)
    incorrect_keys = set(task_values.keys()) - set(allowed_attributes)
    if incorrect_keys:
        raise exception.Invalid('The keys %s are not valid' % str(incorrect_keys))
    task_info_values = _pop_task_info_values(task_values)
    task = _task_format(task_id, **task_values)
    DATA['tasks'][task_id] = task
    task_info = _task_info_create(task['id'], task_info_values)
    return _format_task_from_db(task, task_info)