from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def enable_sra(self):
    if self.module.check_mode:
        self.changed = True
        return
    self.add_proxy_details()
    cmd = 'chsra'
    cmdopts = {}
    cmdargs = ['-enable']
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
    if self.support == 'remote':
        cmdargs = ['-remotesupport', 'enable']
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
    self.log('%s support assistance enabled', self.support.capitalize())
    self.changed = True