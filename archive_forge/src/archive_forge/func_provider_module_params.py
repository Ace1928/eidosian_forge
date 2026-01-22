from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def provider_module_params(self):
    provider_params = [(key, value) for key, value in self._module.params.items() if key not in self.non_provider_params]
    provider_params.append(('data_center', self.get_data_center()))
    return provider_params