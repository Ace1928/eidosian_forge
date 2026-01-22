from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_active_directory_rest(self):
    dummy, error = rest_generic.delete_async(self.rest_api, 'protocols/active-directory', self.svm_uuid, body={'username': self.parameters['admin_username'], 'password': self.parameters['admin_password']})
    if error:
        self.module.fail_json(msg='Error deleting vserver Active Directory %s: %s' % (self.parameters['account_name'], to_native(error)), exception=traceback.format_exc())