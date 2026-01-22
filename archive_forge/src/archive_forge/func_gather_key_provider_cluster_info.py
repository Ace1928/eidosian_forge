from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
@staticmethod
def gather_key_provider_cluster_info(key_provider_clusters):
    key_provider_cluster_facts = []
    if not key_provider_clusters:
        return key_provider_cluster_facts
    for kp_item in key_provider_clusters:
        kp_info = dict(key_provide_id=kp_item.clusterId.id, use_as_default=kp_item.useAsDefault, management_type=kp_item.managementType, has_backup=kp_item.hasBackup, tpm_required=kp_item.tpmRequired, key_id=kp_item.keyId)
        kmip_servers = []
        if hasattr(kp_item, 'servers') and len(kp_item.servers) != 0:
            for kmip_item in kp_item.servers:
                kmip_info = dict(name=kmip_item.name, address=kmip_item.address, port=kmip_item.port, protocol=kmip_item.protocol, proxy=kmip_item.proxyAddress, proxy_port=kmip_item.proxyPort, user_name=kmip_item.userName)
                kmip_servers.append(kmip_info)
        kp_info.update(servers=kmip_servers)
        key_provider_cluster_facts.append(kp_info)
    return key_provider_cluster_facts