from __future__ import (absolute_import, division, print_function)
import os
import json
from datetime import datetime
from os.path import exists
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import idrac_redfish_job_tracking, \
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.parse import urlparse
def get_buffer_text(module, share):
    buffer_text = None
    if share['share_type'] == 'LOCAL':
        file_path = '{0}{1}{2}'.format(share['share_name'], os.sep, share['file_name'])
        if not exists(file_path):
            module.fail_json(msg=INVALID_FILE)
        with open(file_path, 'r') as file_obj:
            buffer_text = file_obj.read()
    return buffer_text