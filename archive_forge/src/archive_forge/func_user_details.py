from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
from ansible.module_utils.six.moves import http_client as httplib
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import quote
import logging
def user_details(module):
    username = module.params['username']
    cyberark_session = module.params['cyberark_session']
    api_base_url = cyberark_session['api_base_url']
    validate_certs = cyberark_session['validate_certs']
    result = {}
    end_point = '/PasswordVault/WebServices/PIMServices.svc/Users/{pusername}'.format(pusername=username)
    url = construct_url(api_base_url, end_point)
    headers = {'Content-Type': 'application/json', 'Authorization': cyberark_session['token'], 'User-Agent': 'CyberArk/1.0 (Ansible; cyberark.pas)'}
    try:
        response = open_url(url, method='GET', headers=headers, validate_certs=validate_certs, timeout=module.params['timeout'])
        result = {'result': json.loads(response.read())}
        return (False, result, response.getcode())
    except (HTTPError, httplib.HTTPException) as http_exception:
        if http_exception.code == 404:
            return (False, None, http_exception.code)
        else:
            module.fail_json(msg='Error while performing user_details.Please validate parameters provided.\n*** end_point=%s\n ==> %s' % (url, to_text(http_exception)), headers=headers, status_code=http_exception.code)
    except Exception as unknown_exception:
        module.fail_json(msg='Unknown error while performing user_details.\n*** end_point=%s\n%s' % (url, to_text(unknown_exception)), headers=headers, status_code=-1)