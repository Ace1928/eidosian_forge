from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
import time
def enable_email_callhome(self):
    if self.module.check_mode:
        self.changed = True
        return
    command = 'startemail'
    command_options = {}
    cmdargs = None
    self.restapi.svc_run_command(command, command_options, cmdargs)
    self.log('Email callhome enabled.')