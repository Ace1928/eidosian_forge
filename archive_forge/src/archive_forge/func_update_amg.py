from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils._text import to_native
from ansible.module_utils.urls import open_url
def update_amg(module, ssid, api_url, api_usr, api_pwd, body, amg_id):
    endpoint = 'storage-systems/%s/async-mirrors/%s/role' % (ssid, amg_id)
    url = api_url + endpoint
    post_data = json.dumps(body)
    try:
        request(url, data=post_data, method='POST', url_username=api_usr, url_password=api_pwd, headers=HEADERS)
    except Exception as e:
        module.fail_json(msg='Failed to change role of AMG. Id [%s].  AMG Id [%s].  Error [%s]' % (ssid, amg_id, to_native(e)), exception=traceback.format_exc())
    status_endpoint = 'storage-systems/%s/async-mirrors/%s' % (ssid, amg_id)
    status_url = api_url + status_endpoint
    try:
        rc, status = request(status_url, method='GET', url_username=api_usr, url_password=api_pwd, headers=HEADERS)
    except Exception as e:
        module.fail_json(msg='Failed to check status of AMG after role reversal. Id [%s].  AMG Id [%s].  Error [%s]' % (ssid, amg_id, to_native(e)), exception=traceback.format_exc())
    if 'roleChangeProgress' in status:
        while status['roleChangeProgress'] != 'none':
            try:
                rc, status = request(status_url, method='GET', url_username=api_usr, url_password=api_pwd, headers=HEADERS)
            except Exception as e:
                module.fail_json(msg='Failed to check status of AMG after role reversal. Id [%s].  AMG Id [%s].  Error [%s]' % (ssid, amg_id, to_native(e)), exception=traceback.format_exc())
    return status