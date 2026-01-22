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
def scan_license_codes_for_nlf(self, license_code):
    more_info = 'You %s seeing this error because the original NLF contents were modified by Ansible.  You can use the string filter to keep the original.'
    transformed = False
    original_license_code = license_code
    if "'statusResp'" in license_code:
        license_code, error = self.reformat_nlf(license_code)
        if error:
            error = 'Error: %s  %s' % (error, more_info % 'are')
            self.module.fail_json(msg=error)
        transformed = True
    nlf_dict, is_nlf, error = self.get_nlf_dict(license_code)
    if error and transformed:
        error = 'Error: %s.  Ansible input: %s  %s' % (error, original_license_code, more_info % 'may be')
        self.module.fail_json(msg=error)
    if error:
        msg = 'The license ' + ('will be installed without checking for idempotency.' if self.parameters['state'] == 'present' else 'cannot be removed.')
        msg += '  You are seeing this warning because ' + error
        self.module.warn(msg)
    return (license_code, nlf_dict, is_nlf)