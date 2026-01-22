from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_krbrealm_rest(self):
    api = 'protocols/nfs/kerberos/realms'
    params = {'name': self.parameters['realm'], 'svm.name': self.parameters['vserver'], 'fields': 'kdc,ad_server,svm,comment,password_server,admin_server,clock_skew'}
    record, error = rest_generic.get_one_record(self.rest_api, api, params)
    if error:
        self.module.fail_json(msg='Error fetching kerberos realm %s: %s' % (self.parameters['realm'], to_native(error)))
    if record:
        self.svm_uuid = record['svm']['uuid']
        return {'kdc_ip': self.na_helper.safe_get(record, ['kdc', 'ip']), 'kdc_port': self.na_helper.safe_get(record, ['kdc', 'port']), 'kdc_vendor': self.na_helper.safe_get(record, ['kdc', 'vendor']), 'ad_server_ip': self.na_helper.safe_get(record, ['ad_server', 'address']), 'ad_server_name': self.na_helper.safe_get(record, ['ad_server', 'name']), 'comment': self.na_helper.safe_get(record, ['comment']), 'pw_server_ip': self.na_helper.safe_get(record, ['password_server', 'address']), 'pw_server_port': str(self.na_helper.safe_get(record, ['password_server', 'port'])), 'admin_server_ip': self.na_helper.safe_get(record, ['admin_server', 'address']), 'admin_server_port': str(self.na_helper.safe_get(record, ['admin_server', 'port'])), 'clock_skew': str(self.na_helper.safe_get(record, ['clock_skew']))}
    return None