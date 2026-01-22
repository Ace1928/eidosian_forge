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
def upload_ssl_key(module, idrac, actions, ssl_key, res_id):
    if not os.path.exists(ssl_key) or os.path.isdir(ssl_key):
        module.exit_json(msg=f'Unable to locate the SSL key file at {ssl_key}.', failed=True)
    try:
        with open(ssl_key, 'r') as file:
            scert_file = file.read()
    except OSError as err:
        module.exit_json(msg=str(err), failed=True)
    if not module.check_mode:
        upload_url = actions.get('#DelliDRACCardService.UploadSSLKey')
        if not upload_url:
            module.exit_json('Upload of SSL key not supported', failed=True)
        payload = {}
        payload = {'SSLKeyString': scert_file}
        idrac.invoke_request(upload_url.format(res_id=res_id), 'POST', data=payload)