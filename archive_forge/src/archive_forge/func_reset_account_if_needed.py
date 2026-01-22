from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.six.moves.http_client import HTTPException
import json
import logging
def reset_account_if_needed(module, existing_account):
    cyberark_session = module.params['cyberark_session']
    api_base_url = cyberark_session['api_base_url']
    validate_certs = cyberark_session['validate_certs']
    management_action = deep_get(module.params, 'secret_management.management_action', 'NOT_FOUND', False)
    cpm_new_secret = deep_get(module.params, 'secret_management.new_secret', 'NOT_FOUND', False)
    logging.debug('management_action: %s  cpm_new_secret: %s', management_action, cpm_new_secret)
    result = {}
    end_point = None
    payload = {}
    existing_account_id = None
    if existing_account is not None:
        existing_account_id = existing_account['id']
    elif module.check_mode:
        existing_account_id = 9999
    if management_action == 'change' and cpm_new_secret is not None and (cpm_new_secret != 'NOT_FOUND'):
        logging.debug('CPM change secret for next CPM cycle')
        end_point = '/PasswordVault/API/Accounts/%s/SetNextPassword' % existing_account_id
        payload['ChangeImmediately'] = False
        payload['NewCredentials'] = cpm_new_secret
    elif management_action == 'change_immediately' and (cpm_new_secret == 'NOT_FOUND' or cpm_new_secret is None):
        logging.debug('CPM change_immediately with random secret')
        end_point = '/PasswordVault/API/Accounts/%s/Change' % existing_account_id
        payload['ChangeEntireGroup'] = True
    elif management_action == 'change_immediately' and (cpm_new_secret is not None and cpm_new_secret != 'NOT_FOUND'):
        logging.debug('CPM change immediately secret for next CPM cycle')
        end_point = '/PasswordVault/API/Accounts/%s/SetNextPassword' % existing_account_id
        payload['ChangeImmediately'] = True
        payload['NewCredentials'] = cpm_new_secret
    elif management_action == 'reconcile':
        logging.debug('CPM reconcile secret')
        end_point = '/PasswordVault/API/Accounts/%s/Reconcile' % existing_account_id
    elif 'new_secret' in list(module.params.keys()) and module.params['new_secret'] is not None:
        logging.debug('Change Credential in Vault')
        end_point = '/PasswordVault/API/Accounts/%s/Password/Update' % existing_account_id
        payload['ChangeEntireGroup'] = True
        payload['NewCredentials'] = module.params['new_secret']
    if end_point is not None:
        if module.check_mode:
            logging.debug('Proceeding with Credential Rotation (CHECK_MODE)')
            return (True, result, -1)
        else:
            logging.debug('Proceeding with Credential Rotation')
            result = {'result': None}
            headers = {'Content-Type': 'application/json', 'Authorization': cyberark_session['token'], 'User-Agent': 'CyberArk/1.0 (Ansible; cyberark.pas)'}
            HTTPMethod = 'POST'
            try:
                response = open_url(api_base_url + end_point, method=HTTPMethod, headers=headers, data=json.dumps(payload), validate_certs=validate_certs)
                return (True, result, response.getcode())
            except (HTTPError, HTTPException) as http_exception:
                if isinstance(http_exception, HTTPError):
                    res = json.load(http_exception)
                else:
                    res = to_text(http_exception)
                module.fail_json(msg='Error while performing reset_account.Please validate parameters provided.\n*** end_point=%s%s\n ==> %s' % (api_base_url, end_point, res), headers=headers, payload=payload, status_code=http_exception.code)
            except Exception as unknown_exception:
                module.fail_json(msg='Unknown error while performing delete_account.\n*** end_point=%s%s\n%s' % (api_base_url, end_point, to_text(unknown_exception)), headers=headers, payload=payload, status_code=-1)
    else:
        return (False, result, -1)