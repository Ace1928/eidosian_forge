from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible.module_utils._text import to_native
def volume_creation_parameter_validation(self):
    if self.enable_cloud_snapshot in {True, False}:
        self.module.fail_json(msg='Following parameter not applicable for creation: enable_cloud_snapshot')
    if self.cloud_account_name:
        self.module.fail_json(msg='Following parameter not applicable for creation: cloud_account_name')
    if self.old_name:
        self.module.fail_json(msg='Parameter [old_name] is not supported during volume creation.')
    missing = [item[0] for item in [('pool', self.pool), ('size', self.size)] if not item[1]]
    if missing:
        self.module.fail_json(msg='Missing required parameter while creating: [{0}]'.format(', '.join(missing)))