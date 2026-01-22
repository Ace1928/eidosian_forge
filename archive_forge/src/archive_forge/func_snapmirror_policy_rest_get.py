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
def snapmirror_policy_rest_get(self, policy_name, svm_name):
    """
        get policy type
        There is a set of system level policies, and users can create their own for a SVM
        REST does not return a svm entry for system policies
        svm_name may not exist yet as it can be created when creating the snapmirror relationship
        """
    policy_type = None
    system_policy_type = None
    api = 'snapmirror/policies'
    query = {'name': policy_name, 'fields': 'svm.name,type'}
    records, error = rest_generic.get_0_or_more_records(self.rest_api, api, query)
    if error is None and records is not None:
        for record in records:
            if 'svm' in record:
                if record['svm']['name'] == svm_name:
                    policy_type = record['type']
                    break
            else:
                system_policy_type = record['type']
    if policy_type is None:
        policy_type = system_policy_type
    return (policy_type, error)