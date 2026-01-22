from __future__ import absolute_import, division, print_function
import uuid
import time
import base64
import json
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.cloudmanager.plugins.module_utils.netapp import CloudManagerRestAPI
def deploy_gcp_vm(self, proxy_certificates):
    """
        deploy GCP VM
        """
    response, client_id, error = self.get_custom_data_for_gcp(proxy_certificates)
    if error is not None:
        self.module.fail_json(msg='Error: Not able to get user data for GCP: %s, %s' % (str(error), str(response)))
    user_data = response
    gcp_custom_data = base64.b64encode(user_data.encode())
    gcp_sa_scopes = ['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/compute.readonly', 'https://www.googleapis.com/auth/ndev.cloudman', 'https://www.googleapis.com/auth/ndev.cloudman.readonly']
    tags = []
    if self.parameters['firewall_tags'] is True:
        tags = {'items': ['firewall-tag-bvsu', 'http-server', 'https-server']}
    device_name = self.parameters['name'] + '-vm-disk-boot'
    t = {'name': self.parameters['name'] + '-vm', 'properties': {'disks': [{'autoDelete': True, 'boot': True, 'deviceName': device_name, 'name': device_name, 'source': '\\"$(ref.%s.selfLink)\\"' % device_name, 'type': 'PERSISTENT'}], 'machineType': 'zones/%s/machineTypes/%s' % (self.parameters['zone'], self.parameters['machine_type']), 'metadata': {'items': [{'key': 'serial-port-enable', 'value': 1}, {'key': 'customData', 'value': gcp_custom_data}]}, 'serviceAccounts': [{'email': self.parameters['gcp_service_account_email'], 'scopes': gcp_sa_scopes}], 'tags': tags, 'zone': self.parameters['zone']}, 'metadata': {'dependsOn': [device_name]}, 'type': 'compute.v1.instance'}
    access_configs = []
    if self.parameters['associate_public_ip'] is True:
        access_configs = [{'kind': 'compute#accessConfig', 'name': 'External NAT', 'type': 'ONE_TO_ONE_NAT', 'networkTier': 'PREMIUM'}]
    project_id = self.parameters['project_id']
    if self.parameters.get('network_project_id'):
        project_id = self.parameters['network_project_id']
    t['properties']['networkInterfaces'] = [{'accessConfigs': access_configs, 'kind': 'compute#networkInterface', 'subnetwork': 'projects/%s/regions/%s/subnetworks/%s' % (project_id, self.parameters['region'], self.parameters['subnet_id'])}]
    td = {'name': device_name, 'properties': {'name': device_name, 'sizeGb': 100, 'sourceImage': 'projects/%s/global/images/family/%s' % (self.rest_api.environment_data['GCP_IMAGE_PROJECT'], self.rest_api.environment_data['GCP_IMAGE_FAMILY']), 'type': 'zones/%s/diskTypes/pd-ssd' % self.parameters['zone'], 'zone': self.parameters['zone']}, 'type': 'compute.v1.disks'}
    content = {'resources': [t, td]}
    my_data = str(yaml.dump(content))
    gcp_deployment_template = '{\n  "name": "%s%s",\n  "target": {\n  "config": {\n  "content": "%s"\n  }\n}\n}' % (self.parameters['name'], '-vm-boot-deployment', my_data)
    api_url = GCP_DEPLOYMENT_MANAGER + '/deploymentmanager/v2/projects/%s/global/deployments' % self.parameters['project_id']
    headers = {'X-User-Token': self.rest_api.token_type + ' ' + self.rest_api.gcp_token, 'X-Tenancy-Account-Id': self.parameters['account_id'], 'Authorization': self.rest_api.token_type + ' ' + self.rest_api.gcp_token, 'Content-type': 'application/json', 'Referer': 'Ansible_NetApp', 'X-Agent-Id': self.rest_api.format_client_id(client_id)}
    response, error, dummy = self.rest_api.post(api_url, data=gcp_deployment_template, header=headers, gcp_type=True)
    if error is not None:
        return (response, client_id, error)
    time.sleep(60)
    retries = 16
    while retries > 0:
        agent, error = self.na_helper.get_occm_agent_by_id(self.rest_api, client_id)
        if error is not None:
            self.module.fail_json(msg='Error: Not able to get occm status: %s, %s' % (str(error), str(agent)), client_id=client_id, changed=True)
        if agent['status'] == 'active':
            break
        else:
            time.sleep(30)
        retries -= 1
    if retries == 0:
        msg = 'Connector VM is created and registered.  Taking too long for OCCM agent to be active or not properly setup.'
        msg += '  Latest status: %s' % agent
        self.module.fail_json(msg=msg, client_id=client_id, changed=True)
    return (response, client_id, error)