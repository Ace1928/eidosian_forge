from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
from ansible.module_utils.common import validation
from abc import ABCMeta, abstractmethod
import os.path
import copy
import json
import inspect
import re
def check_task_response_status(self, response, validation_string, data=False):
    """
        Get the site id from the site name.

        Parameters:
            self - The current object details.
            response (dict) - API response.
            validation_string (string) - String used to match the progress status.

        Returns:
            self
        """
    if not response:
        self.msg = 'response is empty'
        self.status = 'exited'
        return self
    if not isinstance(response, dict):
        self.msg = 'response is not a dictionary'
        self.status = 'exited'
        return self
    response = response.get('response')
    if response.get('errorcode') is not None:
        self.msg = response.get('response').get('detail')
        self.status = 'failed'
        return self
    task_id = response.get('taskId')
    while True:
        task_details = self.get_task_details(task_id)
        self.log('Getting task details from task ID {0}: {1}'.format(task_id, task_details), 'DEBUG')
        if task_details.get('isError') is True:
            if task_details.get('failureReason'):
                self.msg = str(task_details.get('failureReason'))
            else:
                self.msg = str(task_details.get('progress'))
            self.status = 'failed'
            break
        if validation_string in task_details.get('progress').lower():
            self.result['changed'] = True
            if data is True:
                self.msg = task_details.get('data')
            self.status = 'success'
            break
        self.log('progress set to {0} for taskid: {1}'.format(task_details.get('progress'), task_id), 'DEBUG')
    return self