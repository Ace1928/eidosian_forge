from __future__ import (absolute_import, division, print_function)
import logging
import time
from ansible.module_utils.basic import missing_required_lib
def wait_on_completion(self, api_url, action_name, task, retries, wait_interval):
    while True:
        cvo_status, failure_error_message, error = self.check_task_status(api_url)
        if error is not None:
            return error
        if cvo_status == -1:
            return 'Failed to %s %s, error: %s' % (task, action_name, failure_error_message)
        elif cvo_status == 1:
            return None
        if retries == 0:
            return 'Taking too long for %s to %s or not properly setup' % (action_name, task)
        time.sleep(wait_interval)
        retries = retries - 1