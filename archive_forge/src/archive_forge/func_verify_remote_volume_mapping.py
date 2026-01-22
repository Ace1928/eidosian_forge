from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def verify_remote_volume_mapping(self):
    self.log('Entering function verify_remote_volume_mapping')
    cmd = 'lsvdiskhostmap'
    cmdargs = {}
    cmdopts = {}
    cmdargs = [self.target_volume]
    remote_hostmap_data = ''
    remote_restapi = self.construct_remote_rest()
    remote_hostmap_data = remote_restapi.svc_obj_info(cmd, cmdopts, cmdargs)
    if remote_hostmap_data:
        self.module.fail_json(msg='The target volume has hostmappings, Migration relationship cannot be created.')