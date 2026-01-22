from __future__ import absolute_import, division, print_function
import re
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
def get_grid_user(self, unique_name):
    api = 'api/v3/grid/users/%s' % unique_name
    response, error = self.rest_api.get(api)
    if error:
        if response['code'] != 404:
            self.module.fail_json(msg=error['text'])
    else:
        return response['data']
    return None