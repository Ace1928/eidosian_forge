from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def set_nioc_resources(self, resources):
    if self.dvs.config.networkResourceControlVersion == 'version3':
        self._update_version3_resources(resources)
    elif self.dvs.config.networkResourceControlVersion == 'version2':
        self._update_version2_resources(resources)