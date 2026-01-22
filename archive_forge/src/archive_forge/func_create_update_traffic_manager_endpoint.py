from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
def create_update_traffic_manager_endpoint(self):
    """
        Creates or updates a Traffic Manager endpoint.

        :return: deserialized Traffic Manager endpoint state dictionary
        """
    self.log('Creating / Updating the Traffic Manager endpoint {0}'.format(self.name))
    parameters = Endpoint(target_resource_id=self.target_resource_id, target=self.target, endpoint_status=self.endpoint_status, weight=self.weight, priority=self.priority, endpoint_location=self.location, min_child_endpoints=self.min_child_endpoints, geo_mapping=self.geo_mapping)
    try:
        response = self.traffic_manager_management_client.endpoints.create_or_update(self.resource_group, self.profile_name, self.type, self.name, parameters)
        return traffic_manager_endpoint_to_dict(response)
    except Exception as exc:
        request_id = exc.request_id if exc.request_id else ''
        self.fail('Error creating the Traffic Manager endpoint {0}, request id {1} - {2}'.format(self.name, request_id, str(exc)))