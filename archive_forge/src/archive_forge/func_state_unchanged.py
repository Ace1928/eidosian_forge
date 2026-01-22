from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible_collections.community.vmware.plugins.module_utils.vmware import connect_to_api
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def state_unchanged(self):
    """Return unchanged state."""
    self.module.exit_json(changed=False)