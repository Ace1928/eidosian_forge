from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
def await_online_quorum_plus_one(self):
    self._exec('rabbitmq-upgrade', ['await_online_quorum_plus_one'])
    self.result['changed'] = True