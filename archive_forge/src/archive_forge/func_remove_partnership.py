from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def remove_partnership(self, location, id):
    if self.module.check_mode:
        self.changed = True
        return
    rest_api = None
    cmd = 'rmpartnership'
    if location == 'local':
        rest_api = self.restapi_local
    if location == 'remote':
        rest_api = self.restapi_remote
    rest_api.svc_run_command(cmd, {}, [id])
    self.log('Deleted partnership with name %s.', id)
    self.changed = True