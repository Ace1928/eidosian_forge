from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import text_type
from ansible_collections.community.vmware.plugins.plugin_utils.inventory import (
from ansible_collections.community.vmware.plugins.inventory.vmware_vm_inventory import BaseVMwareInventory
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.parsing.yaml.objects import AnsibleVaultEncryptedUnicode
@staticmethod
def get_management_ip(host):
    try:
        vnic_manager = host.configManager.virtualNicManager
        net_config = vnic_manager.QueryNetConfig('management')
        for nic in net_config.candidateVnic:
            if nic.key in net_config.selectedVnic:
                return nic.spec.ip.ipAddress
    except Exception:
        return ''
    return ''