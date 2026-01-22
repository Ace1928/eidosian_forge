from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
def get_grid_gateway_config(self, gateway_id):
    api = 'api/v3/private/gateway-configs/%s' % gateway_id
    response, error = self.rest_api.get(api)
    if error:
        self.module.fail_json(msg=error)
    gateway = response['data']
    gateway_config = self.get_grid_gateway_server_config(gateway['id'])
    return (gateway, gateway_config)