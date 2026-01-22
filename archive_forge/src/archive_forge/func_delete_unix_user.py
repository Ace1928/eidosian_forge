from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_unix_user(self):
    """
        Deletes an UNIX user from a vserver

        :return: None
        """
    user_delete = netapp_utils.zapi.NaElement.create_node_with_children('name-mapping-unix-user-destroy', **{'user-name': self.parameters['name']})
    try:
        self.server.invoke_successfully(user_delete, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error removing UNIX user %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())