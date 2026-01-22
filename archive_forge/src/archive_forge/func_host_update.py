from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def host_update(self, modify, host_data):
    self.log("updating host '%s'", self.name)
    if 'hostcluster' in modify:
        self.addhostcluster()
    elif 'nohostcluster' in modify:
        self.removehostcluster(host_data)
    cmd = 'chhost'
    cmdopts = {}
    if 'fcwwpn' in modify:
        self.host_fcwwpn_update()
        self.changed = True
        self.log('fcwwpn of %s updated', self.name)
    if 'iscsiname' in modify:
        self.host_iscsiname_update()
        self.changed = True
        self.log('iscsiname of %s updated', self.name)
    if 'nqn' in modify:
        self.host_nqn_update()
        self.changed = True
        self.log('nqn of %s updated', self.name)
    if 'type' in modify:
        cmdopts['type'] = self.type
    if 'site' in modify:
        cmdopts['site'] = self.site
    if 'portset' in modify:
        cmdopts['portset'] = self.portset
    if cmdopts:
        cmdargs = [self.name]
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
        self.changed = True
        self.log('type of %s updated', self.name)