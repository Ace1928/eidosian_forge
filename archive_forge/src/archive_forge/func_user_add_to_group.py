from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
from ansible.module_utils.six.moves import http_client as httplib
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import quote
import logging
def user_add_to_group(module):
    username = module.params['username']
    group_name = module.params['group_name']
    vault_id = module.params['vault_id']
    member_type = 'Vault' if module.params['member_type'] is None else module.params['member_type']
    domain_name = module.params['domain_name'] if member_type == 'domain' else None
    cyberark_session = module.params['cyberark_session']
    api_base_url = cyberark_session['api_base_url']
    validate_certs = cyberark_session['validate_certs']
    result = {}
    headers = {'Content-Type': 'application/json', 'Authorization': cyberark_session['token'], 'User-Agent': 'CyberArk/1.0 (Ansible; cyberark.pas)'}
    if group_name and (not vault_id):
        vault_id = resolve_group_name_to_id(module)
        if vault_id is None:
            module.fail_json(msg='Unable to find a user group named {pgroupname}, please create that before adding a user to it'.format(pgroupname=group_name))
    end_point = '/PasswordVault/api/UserGroups/{pvaultid}/Members'.format(pvaultid=vault_id)
    payload = {'memberId': username, 'memberType': member_type}
    if domain_name:
        payload['domainName'] = domain_name
    url = construct_url(api_base_url, end_point)
    try:
        response = open_url(url, method='POST', headers=headers, data=json.dumps(payload), validate_certs=validate_certs, timeout=module.params['timeout'])
        result = {'result': {}}
        return (True, result, response.getcode())
    except (HTTPError, httplib.HTTPException) as http_exception:
        exception_text = to_text(http_exception)
        exception_body = json.loads(http_exception.read().decode())
        if http_exception.code == 409 and ('ITATS262E' in exception_text or exception_body.get('ErrorCode', '') == 'PASWS213E'):
            return (False, None, http_exception.code)
        else:
            module.fail_json(msg='Error while performing user_add_to_group.Please validate parameters provided.\n*** end_point=%s\n ==> %s' % (url, exception_text), payload=payload, headers=headers, status_code=http_exception.code, response=http_exception.read().decode())
    except Exception as unknown_exception:
        module.fail_json(msg='Unknown error while performing user_add_to_group.\n*** end_point=%s\n%s' % (url, to_text(unknown_exception)), payload=payload, headers=headers, status_code=-1)