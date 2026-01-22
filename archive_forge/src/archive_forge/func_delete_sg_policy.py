from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def delete_sg_policy(self):
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'rmsafeguardedpolicy'
    cmdargs = [self.name]
    self.restapi.svc_run_command(cmd, cmdopts=None, cmdargs=cmdargs)
    self.log('Safeguarded policy (%s) deleted', self.name)
    self.changed = True