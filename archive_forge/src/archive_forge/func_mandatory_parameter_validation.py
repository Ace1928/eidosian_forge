from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def mandatory_parameter_validation(self):
    missing = [item[0] for item in [('name', self.name), ('state', self.state)] if not item[1]]
    if missing:
        self.module.fail_json(msg='Missing mandatory parameter: [{0}]'.format(', '.join(missing)))
    if self.volumegroup and self.novolumegroup:
        self.module.fail_json(msg='Mutually exclusive parameters detected: [volumegroup] and [novolumegroup]')