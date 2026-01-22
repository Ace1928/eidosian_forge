from __future__ import absolute_import, division, print_function
import uuid
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi
def check_content_library_status(self):
    """
        Check if Content Library exists or not
        Returns: 'present' if library found, else 'absent'

        """
    ret = 'present' if self.library_name in self.local_libraries else 'absent'
    return ret