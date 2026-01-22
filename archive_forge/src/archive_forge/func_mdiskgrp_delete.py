from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
def mdiskgrp_delete(self):
    if self.module.check_mode:
        self.changed = True
        return
    self.log("deleting mdiskgrp '%s'", self.name)
    cmd = 'rmmdiskgrp'
    cmdopts = None
    cmdargs = [self.name]
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs)