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
def reformat_nlf(self, license_code):
    if not HAS_AST or not HAS_JSON:
        return (None, 'ast and json packages are required to install NLF license files.  Import error(s): %s.' % IMPORT_ERRORS)
    try:
        nlf_dict = ast.literal_eval(license_code)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
        return (None, 'malformed input: %s, exception: %s' % (license_code, exc))
    try:
        license_code = json.dumps(nlf_dict, separators=(',', ':'))
    except Exception as exc:
        return (None, 'unable to encode input: %s - evaluated as %s, exception: %s' % (license_code, nlf_dict, exc))
    return (license_code, None)