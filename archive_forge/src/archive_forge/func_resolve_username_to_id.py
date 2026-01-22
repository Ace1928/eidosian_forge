from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
from ansible.module_utils.six.moves import http_client as httplib
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import quote
import logging
def resolve_username_to_id(module):
    username = module.params['username']
    cyberark_session = module.params['cyberark_session']
    api_base_url = cyberark_session['api_base_url']
    validate_certs = cyberark_session['validate_certs']
    url = construct_url(api_base_url, 'PasswordVault/api/Users?search={pusername}'.format(pusername=username))
    headers = {'Content-Type': 'application/json', 'Authorization': cyberark_session['token'], 'User-Agent': 'CyberArk/1.0 (Ansible; cyberark.pas)'}
    try:
        response = open_url(url, method='GET', headers=headers, validate_certs=validate_certs, timeout=module.params['timeout'])
        users = json.loads(response.read())
        user_id = None
        for user in users['Users']:
            if user['username'] == username:
                if user_id is None:
                    user_id = user['id']
                else:
                    module.fail_json(msg='Found more than one user matching %s, this should be impossible' % username)
        logging.debug('Resolved username {%s} to ID {%s}', username, user_id)
        return user_id
    except (HTTPError, httplib.HTTPException) as http_exception:
        exception_text = to_text(http_exception)
        module.fail_json(msg='Error while performing user_search.Please validate parameters provided.\n*** end_point=%s\n ==> %s' % (url, exception_text), headers=headers, status_code=http_exception.code)
    except Exception as unknown_exception:
        module.fail_json(msg='Unknown error while performing user search.\n*** end_point=%s\n%s' % (url, to_text(unknown_exception)), headers=headers, status_code=-1)