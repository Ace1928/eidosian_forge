from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_aks_kubeconfig(self):
    """
        Gets kubeconfig for the specified AKS instance.

        :return: AKS instance kubeconfig
        """
    access_profile = self.managedcluster_client.managed_clusters.get_access_profile(resource_group_name=self.resource_group, resource_name=self.name, role_name='clusterUser')
    return access_profile.kube_config.decode('utf-8')