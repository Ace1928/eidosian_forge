from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
def delete_ownershipgroup(self):
    if self.module.check_mode:
        self.changed = True
        return
    cmd = 'rmownershipgroup'
    cmdopts = None
    cmdargs = [self.name]
    if self.keepobjects:
        cmdargs.insert(0, '-keepobjects')
    result = self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
    self.changed = True
    self.log('Delete ownership group result: %s', result)