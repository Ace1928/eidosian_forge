from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def vdisk_update(self, modify):
    self.log('Entering function vdisk_update')
    if 'addvdiskcopy' in modify and 'resizevolume' in modify:
        self.module.fail_json(msg='You cannot resize the volume alongwith converting the volume to Standard Mirror')
    if 'addvolumecopy' in modify and 'resizevolume' in modify:
        self.module.fail_json(msg='You cannot resize the volume alongwith converting the volume to Local HyperSwap')
    if 'rmvolumecopy' in modify and 'resizevolume' in modify:
        self.module.fail_json(msg='You cannot resize the volume alongwith converting the Mirror volume to Standard')
    if 'addvolumecopy' in modify:
        self.addvolumecopy()
    elif 'addvdiskcopy' in modify:
        self.isdrpool()
        self.addvdiskcopy()
    elif 'rmvolumecopy' in modify:
        self.rmvolumecopy()
    elif 'resizevolume' in modify:
        self.resizevolume()