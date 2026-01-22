from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def enable_maintenance(self, host):
    if host['resourcestate'] not in ['PrepareForMaintenance', 'Maintenance']:
        self.result['changed'] = True
        args = {'id': host['id']}
        if not self.module.check_mode:
            res = self.query_api('prepareHostForMaintenance', **args)
            self.poll_job(res, 'host')
            host = self._poll_for_maintenance()
    return host