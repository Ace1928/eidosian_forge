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
def get_qos_policy_group(self):
    if self.parameters.get('qos_policy_group') is not None:
        return self.parameters['qos_policy_group']
    if self.parameters.get('qos_adaptive_policy_group') is not None:
        return self.parameters['qos_adaptive_policy_group']
    return None