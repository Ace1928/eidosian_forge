from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def create_remote_hosts(self, hosts_wwpn, hosts_iscsi):
    self.log('Entering function create_remote_hosts()')
    if self.module.check_mode:
        self.changed = True
        return
    remote_hosts_list = []
    source_host_list = []
    remote_hosts_list = self.return_remote_hosts()
    if hosts_iscsi:
        for host, iscsi_vals in hosts_iscsi.items():
            source_host_list.append(host)
    if hosts_wwpn:
        for host, wwpn_vals in hosts_wwpn.items():
            source_host_list.append(host)
    cmd = 'mkhost'
    for host, wwpn in hosts_wwpn.items():
        if host not in remote_hosts_list:
            cmdopts = {'name': host, 'force': True}
            wwpn = ':'.join([str(elem) for elem in wwpn])
            cmdopts['fcwwpn'] = wwpn
            remote_restapi = self.construct_remote_rest()
            remote_restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    for host, iscsi in hosts_iscsi.items():
        if host not in remote_hosts_list:
            cmdopts = {'name': host, 'force': True}
            iscsi = ','.join([str(elem) for elem in iscsi])
            cmdopts['iscsiname'] = iscsi
            remote_restapi = self.construct_remote_rest()
            remote_restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
    if source_host_list:
        self.map_host_vol_remote(source_host_list)