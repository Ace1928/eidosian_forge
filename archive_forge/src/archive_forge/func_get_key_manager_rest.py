from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_key_manager_rest(self):
    api = 'security/key-managers'
    query = {'scope': self.scope}
    fields = 'status,external,uuid,onboard'
    if self.scope == 'svm':
        query['svm.name'] = self.parameters['svm']['name']
    record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
    if error:
        if self.scope == 'svm' and 'SVM "%s" does not exist' % self.parameters['svm']['name'] in error:
            return None
        self.module.fail_json(msg='Error fetching key manager info for %s: %s' % (self.resource, error))
    if record:
        self.uuid = record['uuid']
        if 'external' in record and self.na_helper.safe_get(record, ['onboard', 'enabled']) is False:
            del record['onboard']
        if 'external' in record and 'servers' in record['external']:
            record['external']['servers'] = [{'server': server['server']} for server in record['external']['servers']]
        self.na_helper.remove_hal_links(record)
    return record