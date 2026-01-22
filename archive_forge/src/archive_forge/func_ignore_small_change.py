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
def ignore_small_change(self, current, attribute, threshold):
    if attribute in current and current[attribute] != 0 and (self.parameters.get(attribute) is not None):
        change = abs(current[attribute] - self.parameters[attribute]) * 100.0 / current[attribute]
        if change < threshold:
            self.parameters[attribute] = current[attribute]
            if change > 0.1:
                self.module.warn('resize request for %s ignored: %.1f%% is below the threshold: %.1f%%' % (attribute, change, threshold))