from __future__ import absolute_import, division, print_function
import traceback
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_iscsi_service_rest(self, modify, current):
    if self.use_rest:
        if 'service_state' in modify:
            self.start_or_stop_iscsi_service_rest(modify['service_state'])
        if 'target_alias' in modify:
            self.modify_iscsi_service_state_and_target(modify)
    elif 'service_state' in modify:
        if modify['service_state'] == 'started':
            self.start_iscsi_service()
        else:
            self.stop_iscsi_service()