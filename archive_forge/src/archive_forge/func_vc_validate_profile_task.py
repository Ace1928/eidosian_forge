from __future__ import absolute_import, division, print_function
import_profile:
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
import json
import time
def vc_validate_profile_task(self):
    infra = self.api_client.appliance.infraprofile.Configs
    import_profile_spec = self.get_import_profile_spec()
    validate_task = infra.validate_task(import_profile_spec)
    if 'VALID' == validate_task.get_info().result.get_field('status').value:
        self.module.exit_json(changed=False, status=validate_task.get_info().result.get_field('status').value)
    elif 'INVALID' == validate_task.get_info().result.get_field('status').value:
        self.module.exit_json(changed=False, status=validate_task.get_info().result.get_field('status').value)
    else:
        self.module.fail_json(msg='Failed to validate profile status:"%s" ' % dir(validate_task.get_info().status))