from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
def wait_for_task_completion(module, task):
    """ Poll CVP for the executed task to complete. There is currently no
        timeout. Exits with failure if task status is Failed or Cancelled.

    :param module: Ansible module with parameters and client connection.
    :param task: Task ID to poll for completion.
    :return: true or exit with failure if task is cancelled or fails.
    """
    task_complete = False
    while not task_complete:
        task_info = module.client.api.get_task_by_id(task)
        task_status = task_info['workOrderUserDefinedStatus']
        if task_status == 'Completed':
            return True
        elif task_status in ['Failed', 'Cancelled']:
            module.fail_json(msg=str('Task %s has reported status %s. Please consult the CVP admins for more information.' % (task, task_status)))
        time.sleep(2)