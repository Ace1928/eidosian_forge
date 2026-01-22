from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def return_remote_hosts(self):
    self.log('Entering function return_remote_hosts')
    cmd = 'lshost'
    remote_hosts = []
    cmdopts = {}
    cmdargs = None
    remote_hosts_data = []
    remote_restapi = self.construct_remote_rest()
    remote_hosts_data = remote_restapi.svc_obj_info(cmd, cmdopts, cmdargs)
    self.log(len(remote_hosts_data))
    for host in remote_hosts_data:
        remote_hosts.append(host['name'])
    return remote_hosts