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
def task_get(context, task_id, force_show_deleted=False):
    task, task_info = _task_get(context, task_id, force_show_deleted)
    return _format_task_from_db(task, task_info)