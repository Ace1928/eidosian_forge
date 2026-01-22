from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.common.text.formatters import bytes_to_human
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
def get_datacenter_facts(self):
    datatacenter_facts = {'datacenter': None}
    parent = self.host.parent
    while parent:
        if isinstance(parent, vim.Datacenter):
            datatacenter_facts.update(datacenter=parent.name)
            break
        parent = parent.parent
    return datatacenter_facts