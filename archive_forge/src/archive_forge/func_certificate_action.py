from __future__ import (absolute_import, division, print_function)
import json
import base64
import os
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import iDRACRedfishAPI, idrac_auth_params
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import reset_idrac
def certificate_action(module, idrac, actions, operation, cert_type, res_id):
    cert_url = get_cert_url(actions, operation, cert_type, res_id)
    if not cert_url:
        module.exit_json(msg=NOT_SUPPORTED_ACTION.format(operation=operation, cert_type=module.params.get('certificate_type')))
    cert_payload, method = payload_map.get(cert_type)(module, operation, cert_type)
    exit_certificates(module, idrac, cert_url, cert_payload, method, cert_type, res_id)