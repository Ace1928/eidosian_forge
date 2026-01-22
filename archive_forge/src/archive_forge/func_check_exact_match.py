from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def check_exact_match(self, desired_ports, current_ifgrp):
    matched = set(desired_ports) == set(current_ifgrp)
    if not matched:
        self.rest_api.log_debug(0, 'found LAG with partial match of ports: %s but current is %s' % (desired_ports, current_ifgrp))
    return matched