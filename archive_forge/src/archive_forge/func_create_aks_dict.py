from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_aks_dict(aks):
    """
    Helper method to deserialize a ContainerService to a dict
    :param: aks: ContainerService or AzureOperationPoller with the Azure callback object
    :return: dict with the state on Azure
    """
    return dict(id=aks.id, name=aks.name, location=aks.location, dns_prefix=aks.dns_prefix, kubernetes_version=aks.kubernetes_version, tags=aks.tags, linux_profile=create_linux_profile_dict(aks.linux_profile), service_principal_profile=create_service_principal_profile_dict(aks.service_principal_profile), provisioning_state=aks.provisioning_state, agent_pool_profiles=create_agent_pool_profiles_dict(aks.agent_pool_profiles), type=aks.type, kube_config=aks.kube_config, enable_rbac=aks.enable_rbac, network_profile=create_network_profiles_dict(aks.network_profile), aad_profile=create_aad_profiles_dict(aks.aad_profile), api_server_access_profile=create_api_server_access_profile_dict(aks.api_server_access_profile), addon=create_addon_dict(aks.addon_profiles), fqdn=aks.fqdn, node_resource_group=aks.node_resource_group)