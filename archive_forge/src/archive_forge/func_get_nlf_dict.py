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
def get_nlf_dict(self, license_code):
    nlf_dict = {}
    is_nlf = False
    if '"statusResp"' in license_code:
        if license_code.count('"statusResp"') > 1:
            self.module.fail_json(msg='Error: NLF license files with multiple licenses are not supported, found %d in %s.' % (license_code.count('"statusResp"'), license_code))
        if license_code.count('"serialNumber"') > 1:
            self.module.fail_json(msg='Error: NLF license files with multiple serial numbers are not supported, found %d in %s.' % (license_code.count('"serialNumber"'), license_code))
        is_nlf = True
        if not HAS_JSON:
            return (nlf_dict, is_nlf, 'the json package is required to process NLF license files.  Import error(s): %s.' % IMPORT_ERRORS)
        try:
            nlf_dict = json.loads(license_code)
        except Exception as exc:
            return (nlf_dict, is_nlf, 'the license contents cannot be read.  Unable to decode input: %s - exception: %s.' % (license_code, exc))
    return (nlf_dict, is_nlf, None)