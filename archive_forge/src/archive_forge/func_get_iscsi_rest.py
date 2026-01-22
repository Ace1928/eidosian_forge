from __future__ import absolute_import, division, print_function
import traceback
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_iscsi_rest(self):
    api = 'protocols/san/iscsi/services'
    query = {'svm.name': self.parameters['vserver']}
    fields = 'svm,enabled,target.alias'
    record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg='Error finding iscsi service in %s: %s' % (self.parameters['vserver'], error))
    if record:
        self.uuid = record['svm']['uuid']
        is_started = 'started' if record['enabled'] else 'stopped'
        return {'service_state': is_started, 'target_alias': '' if self.na_helper.safe_get(record, ['target', 'alias']) is None else record['target']['alias']}
    return None