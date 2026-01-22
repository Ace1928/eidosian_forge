from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_upgrades(self, name, resource_group):
    """
        Get supported upgrade version for AKS
        :param: name: str with name of AKS cluster instance
        :param: resource_group: str with resource group containing AKS instance
        :return: dict with available versions for pool profiles and control plane
        """
    cluster = None
    upgrade_profiles = None
    self.log('Get properties for {0}'.format(self.name))
    try:
        cluster = self.managedcluster_client.managed_clusters.get(resource_group_name=resource_group, resource_name=name)
    except ResourceNotFoundError as err:
        self.fail('Error when getting AKS cluster information for {0} : {1}'.format(self.name, err.message or str(err)))
    self.log('Get available upgrade versions for {0}'.format(self.name))
    try:
        upgrade_profiles = self.managedcluster_client.managed_clusters.get_upgrade_profile(resource_group_name=resource_group, resource_name=name)
    except ResourceNotFoundError as err:
        self.fail('Error when getting upgrade versions for {0} : {1}'.format(self.name, err.message or str(err)))
    return dict(agent_pool_profiles=[self.parse_profile(profile) if profile.upgrades else self.default_profile(cluster) for profile in upgrade_profiles.agent_pool_profiles] if upgrade_profiles.agent_pool_profiles else None, control_plane_profile=self.parse_profile(upgrade_profiles.control_plane_profile) if upgrade_profiles.control_plane_profile.upgrades else self.default_profile(cluster))