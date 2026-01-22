from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def remove_snapshot(self):
    snapshot = self.get_snapshot()
    if snapshot:
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('deleteVMSnapshot', vmsnapshotid=snapshot['id'])
            poll_async = self.module.params.get('poll_async')
            if res and poll_async:
                res = self.poll_job(res, 'vmsnapshot')
    return snapshot