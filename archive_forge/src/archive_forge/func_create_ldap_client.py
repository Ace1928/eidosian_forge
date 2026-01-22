from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def create_ldap_client(self):
    """
        Create LDAP client configuration
        """
    options = {'ldap-client-config': self.parameters['name'], 'schema': self.parameters['schema']}
    for attribute in self.simple_attributes:
        if self.parameters.get(attribute) is not None:
            options[str(attribute).replace('_', '-')] = str(self.parameters[attribute])
    ldap_client_create = netapp_utils.zapi.NaElement.create_node_with_children('ldap-client-create', **options)
    if self.parameters.get('servers') is not None:
        self.add_element_with_children('ldap-servers', 'servers', 'string', ldap_client_create)
    if self.parameters.get('preferred_ad_servers') is not None:
        self.add_element_with_children('preferred-ad-servers', 'preferred_ad_servers', 'ip-address', ldap_client_create)
    try:
        self.server.invoke_successfully(ldap_client_create, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as errcatch:
        self.module.fail_json(msg='Error creating LDAP client %s: %s' % (self.parameters['name'], to_native(errcatch)), exception=traceback.format_exc())