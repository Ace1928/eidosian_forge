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
def get_svm_from_destination_vserver_or_path(self):
    svm_name = self.parameters.get('destination_vserver')
    if svm_name is None:
        path = self.parameters.get('destination_path')
        if path is not None:
            svm_name = path.split(':', 1)[0]
    return svm_name