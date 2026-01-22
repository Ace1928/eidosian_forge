from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils.basic import AnsibleModule
from traceback import format_exc
def existing_vdisk(self, volname):
    merged_result = {}
    data = self.restapi.svc_obj_info(cmd='lsvdisk', cmdopts={'bytes': True}, cmdargs=[volname])
    if not data:
        self.log('source volume %s does not exist', volname)
        return
    if isinstance(data, list):
        for d in data:
            merged_result.update(d)
    else:
        merged_result = data
    return merged_result