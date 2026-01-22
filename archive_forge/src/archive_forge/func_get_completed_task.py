import logging
from oslo_concurrency import lockutils
from oslo_context import context
from oslo_utils import excutils
from oslo_utils import reflection
from oslo_vmware._i18n import _
from oslo_vmware.common import loopingcall
from oslo_vmware import exceptions
from oslo_vmware import pbm
from oslo_vmware import vim
from oslo_vmware import vim_util
def get_completed_task():
    complete_time = getattr(task_info, 'completeTime', None)
    if complete_time:
        duration = complete_time - task_info.queueTime
        task_detail['duration_secs'] = duration.total_seconds()
    return task_detail