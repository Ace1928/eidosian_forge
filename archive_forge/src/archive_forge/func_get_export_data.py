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
def get_export_data(idrac, cert_type, res_id):
    try:
        resp = idrac.invoke_request(EXPORT_SSL.format(res_id=res_id), 'POST', data={'SSLCertType': cert_type})
        cert_data = resp.json_data
    except Exception:
        cert_data = {'CertificateFile': ''}
    return cert_data.get('CertificateFile')