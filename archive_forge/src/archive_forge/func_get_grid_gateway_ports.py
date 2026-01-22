from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
def get_grid_gateway_ports(self, target_port):
    configured_ports = []
    gateway = {}
    gateway_config = {}
    api = 'api/v3/private/gateway-configs'
    response, error = self.rest_api.get(api)
    if error:
        self.module.fail_json(msg=error)
    grid_gateway_ports = response['data']
    configured_ports = [data['port'] for data in grid_gateway_ports]
    for index, port in enumerate(configured_ports):
        if target_port == port and grid_gateway_ports[index]['displayName'] == self.parameters['display_name']:
            gateway = grid_gateway_ports[index]
            gateway_config = self.get_grid_gateway_server_config(gateway['id'])
            break
    return (gateway, gateway_config)