from __future__ import absolute_import, division, print_function
import re
import sys
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def get_sn_and_product(self, nlf_dict):
    n_serial_number = self.na_helper.safe_get(nlf_dict, ['statusResp', 'serialNumber']) or self.na_helper.safe_get(nlf_dict, ['statusResp', 'licenses', 'serialNumber'])
    n_product = self.na_helper.safe_get(nlf_dict, ['statusResp', 'product']) or self.na_helper.safe_get(nlf_dict, ['statusResp', 'licenses', 'product'])
    return (n_serial_number, n_product)