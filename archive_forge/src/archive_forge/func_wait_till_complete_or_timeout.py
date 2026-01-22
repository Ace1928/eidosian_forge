from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec,
from re import compile, match, sub
from time import sleep
def wait_till_complete_or_timeout(self, node_name, task_id):
    timeout = self.module.params['timeout']
    while timeout:
        if self.api_task_ok(node_name, task_id):
            return True
        timeout -= 1
        if timeout <= 0:
            return False
        sleep(1)