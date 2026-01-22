from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
def update_identity_federation(self, test=False):
    api = 'api/v3/org/identity-source'
    params = {}
    if test:
        params['test'] = True
    response, error = self.rest_api.put(api, self.data, params=params)
    if error:
        self.module.fail_json(msg=error, payload=self.data)
    if response is not None:
        return response['data']
    else:
        return None