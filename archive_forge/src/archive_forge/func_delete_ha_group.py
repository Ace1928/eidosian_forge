from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
def delete_ha_group(self, ha_group_id):
    api = 'api/v3/private/ha-groups/%s' % ha_group_id
    dummy, error = self.rest_api.delete(api, self.data)
    if error:
        self.module.fail_json(msg=error)