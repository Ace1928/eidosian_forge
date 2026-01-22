from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_aks(self):
    """
        Gets the properties of the specified container service.

        :return: deserialized AKS instance state dictionary
        """
    self.log('Checking if the AKS instance {0} is present'.format(self.name))
    try:
        response = self.managedcluster_client.managed_clusters.get(self.resource_group, self.name)
        self.log('Response : {0}'.format(response))
        self.log('AKS instance : {0} found'.format(response.name))
        response.kube_config = self.get_aks_kubeconfig()
        return create_aks_dict(response)
    except ResourceNotFoundError:
        self.log('Did not find the AKS instance.')
        return False