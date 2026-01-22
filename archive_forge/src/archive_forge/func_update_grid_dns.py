from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
def update_grid_dns(self):
    api = 'api/v3/grid/dns-servers'
    response, error = self.rest_api.put(api, self.data)
    if error:
        self.module.fail_json(msg=error)
    return response['data']