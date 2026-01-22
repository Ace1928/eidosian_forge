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
def convert_general_params(self, module):
    body = {}
    if module.params['eku']:
        body['eku'] = module.params['eku']
    if self.request_type == 'new':
        body['certType'] = module.params['cert_type']
    body['clientId'] = module.params['client_id']
    body.update(convert_module_param_to_json_bool(module, 'ctLog', 'ct_log'))
    body.update(convert_module_param_to_json_bool(module, 'endUserKeyStorageAgreement', 'end_user_key_storage_agreement'))
    return body