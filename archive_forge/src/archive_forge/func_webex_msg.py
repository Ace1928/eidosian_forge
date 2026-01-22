from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def webex_msg(module):
    """When check mode is specified, establish a read only connection, that does not return any user specific
    data, to validate connectivity. In regular mode, send a message to a Cisco Webex Teams Room or Individual"""
    results = {}
    ansible = module.params
    headers = {'Authorization': 'Bearer {0}'.format(ansible['personal_token']), 'content-type': 'application/json'}
    if module.check_mode:
        url = 'https://webexapis.com/v1/people/me'
        payload = None
    else:
        url = 'https://webexapis.com/v1/messages'
        payload = {ansible['recipient_type']: ansible['recipient_id'], ansible['msg_type']: ansible['msg']}
        payload = module.jsonify(payload)
    response, info = fetch_url(module, url, data=payload, headers=headers)
    status_code = info['status']
    msg = info['msg']
    if status_code != 200:
        results['failed'] = True
        results['status_code'] = status_code
        results['message'] = msg
    else:
        results['failed'] = False
        results['status_code'] = status_code
        if module.check_mode:
            results['message'] = 'Authentication Successful.'
        else:
            results['message'] = msg
    return results