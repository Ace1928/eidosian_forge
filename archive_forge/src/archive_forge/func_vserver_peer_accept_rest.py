from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def vserver_peer_accept_rest(self, target):
    vserver_peer_info = self.vserver_peer_get_rest('peer')
    if not vserver_peer_info:
        self.module.fail_json(msg='Error reading vserver peer information on peer %s' % self.parameters['peer_vserver'])
    api = 'svm/peers'
    body = {'state': 'peered'}
    if 'local_name_for_source' in self.parameters:
        body['name'] = self.parameters['local_name_for_source']
    dummy, error = rest_generic.patch_async(self.dst_rest_api, api, vserver_peer_info['local_peer_vserver_uuid'], body)
    self.check_and_report_rest_error(error, 'accepting', self.parameters['peer_vserver'])