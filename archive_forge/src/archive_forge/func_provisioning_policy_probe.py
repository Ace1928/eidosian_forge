from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def provisioning_policy_probe(self):
    updates = []
    self.rename_validation(updates)
    if self.capacitysaving:
        capsav = 'none' if self.capacitysaving == 'drivebased' else self.capacitysaving
        if capsav and capsav != self.pp_data.get('capacity_saving', ''):
            self.module.fail_json(msg='Following paramter not applicable for update operation: capacitysaving')
    if self.deduplicated and (not strtobool(self.pp_data.get('deduplicated', 0))):
        self.module.fail_json(msg='Following paramter not applicable for update operation: deduplicated')
    return updates