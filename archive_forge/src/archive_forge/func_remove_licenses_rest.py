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
def remove_licenses_rest(self, package_name, nlf_dict):
    """
        This is called either with a package name or a NLF dict
        We already validated product and serialNumber are present in nlf_dict
        """
    p_serial_number = self.parameters.get('serial_number')
    n_serial_number = nlf_dict.get('serialNumber')
    n_product = nlf_dict.get('product')
    serial_number = n_serial_number or p_serial_number
    if not serial_number:
        self.module.fail_json(msg='Error: serial_number is required to delete a license.')
    if n_product:
        error = self.remove_one_license_rest(None, n_product, serial_number)
    elif package_name.endswith(('Bundle', 'Edition')):
        error = self.remove_one_license_rest(None, package_name, serial_number)
    else:
        error = self.remove_one_license_rest(package_name, None, serial_number)
        if error and "entry doesn't exist" in error:
            return False
    if error:
        self.module.fail_json(msg='Error removing license for serial number %s and %s: %s' % (serial_number, n_product or package_name, error))
    return True