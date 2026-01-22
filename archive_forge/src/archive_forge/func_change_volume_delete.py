from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def change_volume_delete(self):
    cmd = 'rmvolume'
    cmdopts = None
    cmdargs = [self.cvname]
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
    self.changed = True
    self.log('Delete vdisk %s', self.cvname)