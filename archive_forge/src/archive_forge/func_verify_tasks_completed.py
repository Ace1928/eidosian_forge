from __future__ import absolute_import, division, print_function
import json
import os
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.urls import url_argument_spec
def verify_tasks_completed(self, response_dict):
    for task in response_dict['spawned_tasks']:
        task_url = '%s%s' % (self.host, task['_href'])
        while True:
            response, info = fetch_url(self.module, task_url, data='', method='GET')
            if info['status'] != 200:
                self.module.fail_json(msg='Failed to check async task status.', status_code=info['status'], response=info['msg'], url=task_url)
            task_dict = json.load(response)
            if task_dict['state'] == 'finished':
                return True
            if task_dict['state'] == 'error':
                self.module.fail_json(msg='Asynchronous task failed to complete.', error=task_dict['error'])
            sleep(2)