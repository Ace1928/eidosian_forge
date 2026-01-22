from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.aws.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.aws.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.aws.plugins.module_utils.netapp import AwsCvsRestAPI
def delete_activedirectory(self):
    activedirectory_id = self.get_activedirectory_id()
    if activedirectory_id:
        api = 'Storage/ActiveDirectory/' + activedirectory_id
        data = None
        response, error = self.rest_api.delete(api, data)
        if not error:
            return response
        else:
            self.module.fail_json(msg=response['message'])
    else:
        self.module.fail_json(msg='Active Directory does not exist')