from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def filter_get_results(self, record):
    record['rule_index'] = record.pop('index')
    record['anonymous_user_id'] = record.pop('anonymous_user')
    record['protocol'] = record.pop('protocols')
    record['super_user_security'] = record.pop('superuser')
    record['client_match'] = [each['match'] for each in record['clients']]
    record.pop('clients')
    return record