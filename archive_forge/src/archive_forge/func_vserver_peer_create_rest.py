from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def vserver_peer_create_rest(self):
    """
        Create a vserver peer using rest
        """
    api = 'svm/peers'
    query = {'return_records': 'true'}
    body = {'svm.name': self.parameters['vserver'], 'peer.cluster.name': self.parameters['peer_cluster'], 'peer.svm.name': self.parameters['peer_vserver'], 'applications': self.parameters['applications']}
    if 'local_name_for_peer' in self.parameters:
        body['name'] = self.parameters['local_name_for_peer']
    record, error = rest_generic.post_async(self.rest_api, api, body, query)
    self.check_and_report_rest_error(error, 'creating', self.parameters['vserver'])
    if record.get('records') is not None:
        self.peer_relation_uuid = record['records'][0].get('uuid')