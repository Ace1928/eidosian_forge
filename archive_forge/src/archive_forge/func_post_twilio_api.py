from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url
def post_twilio_api(module, account_sid, auth_token, msg, from_number, to_number, media_url=None):
    URI = 'https://api.twilio.com/2010-04-01/Accounts/%s/Messages.json' % (account_sid,)
    AGENT = 'Ansible'
    data = {'From': from_number, 'To': to_number, 'Body': msg}
    if media_url:
        data['MediaUrl'] = media_url
    encoded_data = urlencode(data)
    headers = {'User-Agent': AGENT, 'Content-type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'}
    module.params['url_username'] = account_sid.replace('\n', '')
    module.params['url_password'] = auth_token.replace('\n', '')
    return fetch_url(module, URI, data=encoded_data, headers=headers)