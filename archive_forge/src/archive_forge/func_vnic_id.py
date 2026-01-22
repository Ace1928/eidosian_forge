from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
@vnic_id.setter
def vnic_id(self, vnic_id):
    self._vnic_id = vnic_id