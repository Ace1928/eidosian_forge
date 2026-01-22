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
def remove_one_license_rest(self, package_name, product, serial_number):
    api = 'cluster/licensing/licenses'
    query = {'serial_number': serial_number}
    if product:
        query['licenses.installed_license'] = product.replace(' ', '*')
        query['state'] = '*'
    dummy, error = rest_generic.delete_async(self.rest_api, api, package_name, query)
    return error