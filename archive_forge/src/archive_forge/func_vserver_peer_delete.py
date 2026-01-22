from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def vserver_peer_delete(self, current):
    """
        Delete a vserver peer
        """
    if self.use_rest:
        return self.vserver_peer_delete_rest(current)
    vserver_peer_delete = netapp_utils.zapi.NaElement.create_node_with_children('vserver-peer-delete', **{'peer-vserver': current['local_peer_vserver'], 'vserver': self.parameters['vserver']})
    try:
        self.server.invoke_successfully(vserver_peer_delete, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting vserver peer %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())