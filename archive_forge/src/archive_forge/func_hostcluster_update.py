from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def hostcluster_update(self, modify):
    if self.module.check_mode:
        self.changed = True
        return
    self.log("updating host cluster '%s'", self.name)
    cmd = 'chhostcluster'
    cmdopts = {}
    if 'ownershipgroup' in modify:
        cmdopts['ownershipgroup'] = self.ownershipgroup
    elif 'noownershipgroup' in modify:
        cmdopts['noownershipgroup'] = self.noownershipgroup
    if cmdopts:
        cmdargs = [self.name]
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        self.changed = True
        self.log('Properties of %s updated', self.name)