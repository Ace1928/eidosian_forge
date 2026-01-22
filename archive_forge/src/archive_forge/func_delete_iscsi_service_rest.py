from __future__ import absolute_import, division, print_function
import traceback
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_iscsi_service_rest(self, current):
    if current['service_state'] == 'started':
        self.start_or_stop_iscsi_service_rest('stopped')
    api = 'protocols/san/iscsi/services'
    dummy, error = rest_generic.delete_async(self.rest_api, api, self.uuid)
    if error:
        self.module.fail_json(msg='Error deleting iscsi service on vserver %s: %s' % (self.parameters['vserver'], error))