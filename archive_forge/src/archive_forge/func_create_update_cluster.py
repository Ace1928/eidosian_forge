from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def create_update_cluster(self):
    """
        Creates or updates Cluster with the specified configuration.

        :return: deserialized Cluster instance state dictionary
        """
    self.log('Creating / Updating the Cluster instance {0}'.format(self.name))
    try:
        if self.to_do == Actions.Create:
            response = self.mgmt_client.clusters.begin_create(resource_group_name=self.resource_group, cluster_name=self.name, parameters=self.parameters)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        else:
            if self.tags_changed:
                response = self.mgmt_client.clusters.update(resource_group_name=self.resource_group, cluster_name=self.name, parameters={'tags': self.parameters.get('tags')})
                if isinstance(response, LROPoller):
                    response = self.get_poller_result(response)
            if self.new_instance_count:
                response = self.mgmt_client.clusters.begin_resize(resource_group_name=self.resource_group, cluster_name=self.name, role_name='workernode', parameters={'target_instance_count': self.new_instance_count})
                if isinstance(response, LROPoller):
                    response = self.get_poller_result(response)
    except Exception as exc:
        self.fail('Error creating or updating Cluster instance: {0}'.format(str(exc)))
    return response.as_dict() if response else {}