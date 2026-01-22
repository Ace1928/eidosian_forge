from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_active_directory(self):
    if self.use_rest:
        return self.delete_active_directory_rest()
    active_directory_obj = netapp_utils.zapi.NaElement('active-directory-account-delete')
    active_directory_obj.add_new_child('admin-password', self.parameters['admin_password'])
    active_directory_obj.add_new_child('admin-username', self.parameters['admin_username'])
    try:
        self.server.invoke_successfully(active_directory_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting vserver Active Directory %s: %s' % (self.parameters['account_name'], to_native(error)))