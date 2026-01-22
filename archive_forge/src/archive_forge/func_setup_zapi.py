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
def setup_zapi(self):
    if self.parameters.get('identity_preservation'):
        self.module.fail_json(msg='Error: The option identity_preservation is supported only with REST.')
    if not netapp_utils.has_netapp_lib():
        self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
    host_options = self.parameters['peer_options'] if self.parameters.get('connection_type') == 'ontap_elementsw' else None
    return netapp_utils.setup_na_ontap_zapi(module=self.module, host_options=host_options)