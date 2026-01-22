from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils._text import to_native
from ansible.module_utils.urls import open_url
def get_pwd_status(module, ssid, api_url, user, pwd):
    pwd_status = 'storage-systems/%s/passwords' % ssid
    url = api_url + pwd_status
    try:
        rc, data = request(url, headers=HEADERS, url_username=user, url_password=pwd, validate_certs=module.validate_certs)
        return (data['readOnlyPasswordSet'], data['adminPasswordSet'])
    except HTTPError as e:
        module.fail_json(msg='There was an issue with connecting, please check that your endpoint is properly defined and your credentials are correct: %s' % to_native(e))