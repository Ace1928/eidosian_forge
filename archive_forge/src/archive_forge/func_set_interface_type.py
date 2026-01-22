from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
def set_interface_type(self, interface_type):
    if 'interface_type' in self.parameters:
        if self.parameters['interface_type'] != interface_type:
            self.module.fail_json(msg='Error: mismatch between configured interface_type: %s and derived interface_type: %s.' % (self.parameters['interface_type'], interface_type))
    else:
        self.parameters['interface_type'] = interface_type