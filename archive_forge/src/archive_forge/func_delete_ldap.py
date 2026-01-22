from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def delete_ldap(self):
    """
        Delete LDAP configuration
        """
    ldap_client_delete = netapp_utils.zapi.NaElement.create_node_with_children('ldap-config-delete', **{})
    try:
        self.server.invoke_successfully(ldap_client_delete, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as errcatch:
        self.module.fail_json(msg='Error deleting LDAP configuration %s: %s' % (self.parameters['name'], to_native(errcatch)), exception=traceback.format_exc())