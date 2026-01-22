from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
from ansible.module_utils.six.moves import http_client as httplib
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import quote
import logging
def user_delete(module):
    cyberark_session = module.params['cyberark_session']
    api_base_url = cyberark_session['api_base_url']
    validate_certs = cyberark_session['validate_certs']
    result = {}
    vault_user_id = resolve_username_to_id(module)
    if vault_user_id is None:
        return (False, result, None)
    end_point = 'PasswordVault/api/Users/{pvaultuserid}'.format(pvaultuserid=vault_user_id)
    headers = {'Content-Type': 'application/json', 'Authorization': cyberark_session['token'], 'User-Agent': 'CyberArk/1.0 (Ansible; cyberark.pas)'}
    url = construct_url(api_base_url, end_point)
    try:
        response = open_url(url, method='DELETE', headers=headers, validate_certs=validate_certs, timeout=module.params['timeout'])
        result = {'result': {}}
        return (True, result, response.getcode())
    except (HTTPError, httplib.HTTPException) as http_exception:
        exception_text = to_text(http_exception)
        if http_exception.code == 404 and 'ITATS003E' in exception_text:
            result = {'result': {}}
            return (False, result, http_exception.code)
        else:
            module.fail_json(msg='Error while performing user_delete.Please validate parameters provided.\n*** end_point=%s\n ==> %s' % (url, exception_text), headers=headers, status_code=http_exception.code)
    except Exception as unknown_exception:
        module.fail_json(msg='Unknown error while performing user_delete.\n*** end_point=%s\n%s' % (url, to_text(unknown_exception)), headers=headers, status_code=-1)