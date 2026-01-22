from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_cifs_share(self):
    """
        Delete CIFS share
        """
    cifs_delete = netapp_utils.zapi.NaElement.create_node_with_children('cifs-share-delete', **{'share-name': self.parameters.get('name')})
    try:
        self.server.invoke_successfully(cifs_delete, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting cifs-share %s: %s' % (self.parameters.get('name'), to_native(error)), exception=traceback.format_exc())