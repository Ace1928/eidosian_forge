from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def construct_remote_rest(self):
    remote_ip = self.discover_partner_system()
    self.remote_restapi = IBMSVCRestApi(module=self.module, domain='', clustername=remote_ip, username=self.module.params['remote_username'], password=self.module.params['remote_password'], validate_certs=self.module.params['remote_validate_certs'], log_path=self.module.params['log_path'], token=self.module.params['remote_token'])
    return self.remote_restapi