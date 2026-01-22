from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def delete_user_with_auth(self, application, index):
    """
        deletes the user for the given application and authentication_method
        application is now a dict
        :param: application: application to grant access to
        """
    user_delete = netapp_utils.zapi.NaElement.create_node_with_children('security-login-delete', **{'vserver': self.parameters['vserver'], 'user-name': self.parameters['name'], 'application': application['application'], 'authentication-method': application['authentication_methods'][index]})
    try:
        self.server.invoke_successfully(user_delete, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error removing user %s: %s - application: %s' % (self.parameters['name'], to_native(error), application), exception=traceback.format_exc())