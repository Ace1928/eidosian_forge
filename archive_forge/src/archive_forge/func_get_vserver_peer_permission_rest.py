from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_vserver_peer_permission_rest(self):
    """
        Retrieves SVM peer permissions.
        """
    api = 'svm/peer-permissions'
    query = {'svm.name': self.parameters['vserver'], 'cluster_peer.name': self.parameters['cluster_peer'], 'fields': 'svm.uuid,cluster_peer.uuid,applications'}
    record, error = rest_generic.get_one_record(self.rest_api, api, query)
    if error:
        self.module.fail_json(msg='Error on fetching vserver peer permissions: %s' % error)
    if record:
        self.svm_uuid = self.na_helper.safe_get(record, ['svm', 'uuid'])
        self.cluster_peer_uuid = self.na_helper.safe_get(record, ['cluster_peer', 'uuid'])
        return {'applications': self.na_helper.safe_get(record, ['applications'])}
    return None