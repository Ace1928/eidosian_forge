from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def get_aggr_rest(self, name):
    if not name:
        return None
    api = 'storage/aggregates'
    query = {'name': name}
    fields = 'uuid,state,block_storage.primary.disk_count,data_encryption,snaplock_type'
    if 'tags' in self.parameters:
        fields += ',_tags'
    record, error = rest_generic.get_one_record(self.rest_api, api, query, fields)
    if error:
        self.module.fail_json(msg='Error: failed to get aggregate %s: %s' % (name, error))
    if record:
        return {'tags': record.get('_tags', []), 'disk_count': self.na_helper.safe_get(record, ['block_storage', 'primary', 'disk_count']), 'encryption': self.na_helper.safe_get(record, ['data_encryption', 'software_encryption_enabled']), 'service_state': record['state'], 'snaplock_type': record['snaplock_type'], 'uuid': record['uuid']}
    return None