from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlencode, urlunparse
from datetime import datetime
def send_event_v1(module, service_key, event_type, desc, incident_key=None, client=None, client_url=None):
    url = 'https://events.pagerduty.com/generic/2010-04-15/create_event.json'
    headers = {'Content-type': 'application/json'}
    data = {'service_key': service_key, 'event_type': event_type, 'incident_key': incident_key, 'description': desc, 'client': client, 'client_url': client_url}
    response, info = fetch_url(module, url, method='post', headers=headers, data=json.dumps(data))
    if info['status'] != 200:
        module.fail_json(msg='failed to %s. Reason: %s' % (event_type, info['msg']))
    json_out = json.loads(response.read())
    return json_out