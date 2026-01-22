from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def create_provisioning_policy(self):
    self.create_validation()
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'mkprovisioningpolicy'
    cmdopts = {'name': self.name, 'capacitysaving': self.capacitysaving, 'deduplicated': self.deduplicated}
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    self.log('Provisioning policy (%s) created', self.name)
    self.changed = True