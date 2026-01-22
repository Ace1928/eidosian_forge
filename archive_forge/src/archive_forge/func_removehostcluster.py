from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def removehostcluster(self, data):
    if self.module.check_mode:
        self.changed = True
        return
    self.log("removing host '%s' from hostcluster %s", self.name, data['host_cluster_name'])
    hostcluster_name = data['host_cluster_name']
    cmd = 'rmhostclustermember'
    cmdopts = {}
    cmdargs = [hostcluster_name]
    cmdopts['host'] = self.name
    cmdopts['keepmappings'] = True
    self.restapi.svc_run_command(cmd, cmdopts, cmdargs)
    self.changed = True