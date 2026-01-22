from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def restart_vpc(self):
    self.result['changed'] = True
    vpc = self.get_vpc()
    if vpc and (not self.module.check_mode):
        args = {'id': vpc['id'], 'cleanup': self.module.params.get('clean_up')}
        res = self.query_api('restartVPC', **args)
        poll_async = self.module.params.get('poll_async')
        if poll_async:
            self.poll_job(res, 'vpc')
    return vpc