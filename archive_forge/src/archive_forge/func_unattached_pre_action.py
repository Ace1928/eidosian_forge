from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def unattached_pre_action(self, storage_domain):
    dc_name = self.param('data_center')
    if not dc_name:
        dc_name = self._find_attached_datacenter_name(storage_domain.name)
    self._service = self._attached_sds_service(dc_name)
    self._maintenance(storage_domain)