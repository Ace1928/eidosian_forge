from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def vserver_peer_get(self, target='source'):
    """
        Get current vserver peer info
        :return: Dictionary of current vserver peer details if query successful, else return None
        """
    if self.use_rest:
        return self.vserver_peer_get_rest(target)
    vserver_peer_get_iter = self.vserver_peer_get_iter(target)
    vserver_info = {}
    try:
        if target == 'source':
            result = self.server.invoke_successfully(vserver_peer_get_iter, enable_tunneling=True)
        elif target == 'peer':
            result = self.dest_server.invoke_successfully(vserver_peer_get_iter, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching vserver peer %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) > 0:
        vserver_peer_info = result.get_child_by_name('attributes-list').get_child_by_name('vserver-peer-info')
        vserver_info['peer_vserver'] = vserver_peer_info.get_child_content('remote-vserver-name')
        vserver_info['vserver'] = vserver_peer_info.get_child_content('vserver')
        vserver_info['local_peer_vserver'] = vserver_peer_info.get_child_content('peer-vserver')
        vserver_info['peer_state'] = vserver_peer_info.get_child_content('peer-state')
        return vserver_info
    return None