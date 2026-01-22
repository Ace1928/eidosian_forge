from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_cifs_acl(self):
    """
        Delete access control for the given CIFS share/user-group
        """
    options = {'share': self.parameters['share_name'], 'user-or-group': self.parameters['user_or_group']}
    if self.parameters.get('type') is not None:
        options['user-group-type'] = self.parameters['type']
    cifs_acl_delete = netapp_utils.zapi.NaElement.create_node_with_children('cifs-share-access-control-delete', **options)
    try:
        self.server.invoke_successfully(cifs_acl_delete, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting cifs-share-access-control %s: %s' % (self.parameters['share_name'], to_native(error)), exception=traceback.format_exc())