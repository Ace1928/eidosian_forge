from __future__ import absolute_import, division, print_function
from ansible_collections.community.crypto.plugins.module_utils.ecs.api import (
import datetime
import os
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.io import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
def convert_cert_subject_params(self, module):
    body = {}
    if module.params['subject_alt_name']:
        body['subjectAltName'] = module.params['subject_alt_name']
    if module.params['org']:
        body['org'] = module.params['org']
    if module.params['ou']:
        body['ou'] = module.params['ou']
    return body