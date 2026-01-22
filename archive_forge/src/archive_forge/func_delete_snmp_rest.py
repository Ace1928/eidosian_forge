from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
def delete_snmp_rest(self, current):
    api = 'support/snmp/users/' + current['engine_id'] + '/' + self.parameters['snmp_username']
    dummy, error = self.rest_api.delete(api)
    if error:
        self.module.fail_json(msg=error)