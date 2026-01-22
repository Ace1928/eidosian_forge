from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def get_name_service_switch_rest(self):
    record, error = rest_vserver.get_vserver(self.rest_api, self.parameters['vserver'], 'nsswitch,uuid')
    if error:
        self.module.fail_json(msg='Error fetching name service switch info for %s: %s' % (self.parameters['vserver'], to_native(error)))
    if not record:
        self.module.fail_json(msg='Error: Specified vserver %s not found' % self.parameters['vserver'])
    self.svm_uuid = record['uuid']
    database_type = self.na_helper.safe_get(record, ['nsswitch', self.parameters['database_type']])
    return {'sources': database_type if database_type else []}