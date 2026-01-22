from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.storage_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def delete_storage_partition(self):
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'rmpartition'
    cmdopts = {}
    if self.deletenonpreferredmanagementobjects:
        cmdopts['deletenonpreferredmanagementobjects'] = self.deletenonpreferredmanagementobjects
    if self.deletepreferredmanagementobjects:
        cmdopts['deletepreferredmanagementobjects'] = self.deletepreferredmanagementobjects
    self.restapi.svc_run_command(cmd, cmdopts=cmdopts, cmdargs=[self.name])
    self.changed = True