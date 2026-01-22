from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def update_parameter_validation(self):
    if self.state == 'present' and (not self.remote_cluster_id):
        self.module.fail_json(msg='Missing required parameter during updation: remote_cluster_id')