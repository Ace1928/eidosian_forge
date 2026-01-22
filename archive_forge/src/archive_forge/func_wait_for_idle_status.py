from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def wait_for_idle_status(self):
    transferring_time_out = self.parameters['transferring_time_out']
    increment = 30
    if transferring_time_out <= 0:
        return self.snapmirror_get()
    for __ in range(0, transferring_time_out, increment):
        time.sleep(increment)
        current = self.snapmirror_get()
        if current and current['status'] != 'transferring':
            return current
    self.module.warn('SnapMirror relationship is still transferring after %d seconds.' % transferring_time_out)
    return current