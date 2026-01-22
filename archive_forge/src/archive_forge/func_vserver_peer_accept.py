from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def vserver_peer_accept(self):
    """
        Accept a vserver peer at destination
        """
    if self.use_rest:
        return self.vserver_peer_accept_rest('peer')
    vserver_peer_info = self.vserver_peer_get('peer')
    if vserver_peer_info is None:
        self.module.fail_json(msg='Error retrieving vserver peer information while accepting')
    vserver_peer_accept = netapp_utils.zapi.NaElement.create_node_with_children('vserver-peer-accept', **{'peer-vserver': vserver_peer_info['local_peer_vserver'], 'vserver': self.parameters['peer_vserver']})
    if 'local_name_for_source' in self.parameters:
        vserver_peer_accept.add_new_child('local-name', self.parameters['local_name_for_source'])
    try:
        self.dest_server.invoke_successfully(vserver_peer_accept, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error accepting vserver peer %s: %s' % (self.parameters['peer_vserver'], to_native(error)), exception=traceback.format_exc())