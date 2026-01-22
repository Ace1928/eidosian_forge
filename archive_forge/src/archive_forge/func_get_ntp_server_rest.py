from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_ntp_server_rest(self):
    api = 'cluster/ntp/servers'
    options = {'server': self.parameters['server_name'], 'fields': 'server,version,key.id'}
    record, error = rest_generic.get_one_record(self.rest_api, api, options)
    if error:
        self.module.fail_json(msg=error)
    if record:
        return {'server': self.na_helper.safe_get(record, ['server']), 'version': self.na_helper.safe_get(record, ['version']), 'key_id': self.na_helper.safe_get(record, ['key', 'id'])}
    return None