from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def create_aws_fsx(self):
    """ Create AWS FSx """
    json = {'name': self.parameters['name'], 'region': self.parameters['region'], 'workspaceId': self.parameters['workspace_id'], 'credentialsId': self.aws_credentials_id, 'throughputCapacity': self.parameters['throughput_capacity'], 'storageCapacity': {'size': self.parameters['storage_capacity_size'], 'unit': self.parameters['storage_capacity_size_unit']}, 'fsxAdminPassword': self.parameters['fsx_admin_password'], 'primarySubnetId': self.parameters['primary_subnet_id'], 'secondarySubnetId': self.parameters['secondary_subnet_id']}
    if self.parameters.get('tags') is not None:
        tags = []
        for each_tag in self.parameters['tags']:
            tag = {'key': each_tag['tag_key'], 'value': each_tag['tag_value']}
            tags.append(tag)
        json.update({'tags': tags})
    if self.parameters.get('security_group_ids'):
        json.update({'securityGroupIds': self.parameters['security_group_ids']})
    if self.parameters.get('route_table_ids'):
        json.update({'routeTableIds': self.parameters['route_table_ids']})
    if self.parameters.get('kms_key_id'):
        json.update({'kmsKeyId': self.parameters['kms_key_id']})
    if self.parameters.get('minimum_ssd_iops'):
        json.update({'minimumSsdIops': self.parameters['minimum_ssd_iops']})
    if self.parameters.get('endpoint_ip_address_range'):
        json.update({'endpointIpAddressRange': self.parameters['endpoint_ip_address_range']})
    api_url = '/fsx-ontap/working-environments/%s' % self.parameters['tenant_id']
    response, error, dummy = self.rest_api.post(api_url, json, header=self.headers)
    if error is not None:
        self.module.fail_json(msg='Error: unexpected response on creating AWS FSx: %s, %s' % (str(error), str(response)))
    working_environment_id = response['id']
    creation_wait_time = 30
    creation_retry_count = 30
    wait_on_completion_api_url = '/fsx-ontap/working-environments/%s/%s?provider-details=true' % (self.parameters['tenant_id'], working_environment_id)
    err = self.wait_on_completion_for_fsx(wait_on_completion_api_url, 'AWS_FSX', 'create', creation_retry_count, creation_wait_time)
    if err is not None:
        self.module.fail_json(msg='Error: unexpected response wait_on_completion for creating AWS FSX: %s' % str(err))
    return working_environment_id