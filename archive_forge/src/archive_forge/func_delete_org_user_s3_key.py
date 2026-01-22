from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
def delete_org_user_s3_key(self, user_id, access_key):
    api = 'api/v3/org/users/current-user/s3-access-keys'
    if user_id:
        api = 'api/v3/org/users/%s/s3-access-keys/%s' % (user_id, access_key)
    self.data = None
    response, error = self.rest_api.delete(api, self.data)
    if error:
        self.module.fail_json(msg=error)