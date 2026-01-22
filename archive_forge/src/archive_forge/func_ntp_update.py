from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def ntp_update(self, ip):
    cmd = 'chsystem'
    cmdopts = {}
    cmdopts['ntpip'] = ip
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    self.changed = True
    self.log('NTP IP: %s updated', cmdopts)
    if self.ntpip:
        self.message += ' NTP IP [%s] updated.' % self.ntpip