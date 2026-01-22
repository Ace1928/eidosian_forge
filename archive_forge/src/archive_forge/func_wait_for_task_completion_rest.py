from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def wait_for_task_completion_rest(self, api, query, check_state):
    retries = self.parameters['max_wait_time'] // (self.parameters['check_interval'] + 1)
    fail_count = 0
    while retries > 0:
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            if fail_count < 3:
                fail_count += 1
                retries -= 1
                time.sleep(self.parameters['check_interval'])
                continue
            return error
        if record is None:
            return None
        fail_count = 0
        retry_required = check_state(record)
        if not retry_required:
            return None
        time.sleep(self.parameters['check_interval'])
        retries -= 1