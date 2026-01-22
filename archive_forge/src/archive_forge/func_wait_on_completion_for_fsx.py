from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def wait_on_completion_for_fsx(self, api_url, action_name, task, retries, wait_interval):
    while True:
        fsx_status, error = self.check_task_status_for_fsx(api_url)
        if error is not None:
            return error
        if fsx_status['status']['status'] == 'ON' and fsx_status['status']['lifecycle'] == 'AVAILABLE':
            return None
        elif fsx_status['status']['status'] == 'FAILED':
            return 'Failed to %s %s' % (task, action_name)
        if retries == 0:
            return 'Taking too long for %s to %s or not properly setup' % (action_name, task)
        time.sleep(wait_interval)
        retries = retries - 1