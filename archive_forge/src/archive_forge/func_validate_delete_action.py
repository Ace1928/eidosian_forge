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
def validate_delete_action(self, nlf_dict):
    """ make sure product and serialNumber are set at the top level (V2 format) """
    n_serial_number, n_product = self.get_sn_and_product(nlf_dict)
    if nlf_dict and (not n_product):
        self.module.fail_json(msg='Error: product not found in NLF file %s.' % nlf_dict)
    p_serial_number = self.parameters.get('serial_number')
    if p_serial_number and n_serial_number and (p_serial_number != n_serial_number):
        self.module.fail_json(msg='Error: mismatch is serial numbers %s vs %s' % (p_serial_number, n_serial_number))
    if nlf_dict and (not n_serial_number) and (not p_serial_number):
        self.module.fail_json(msg='Error: serialNumber not found in NLF file.  It can be set in the module parameter.')
    nlf_dict['serialNumber'] = n_serial_number or p_serial_number
    nlf_dict['product'] = n_product