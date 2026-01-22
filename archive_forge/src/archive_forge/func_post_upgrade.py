from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
def post_upgrade(self):
    self._exec('rabbitmq-upgrade', ['post_upgrade'])
    self.result['changed'] = True