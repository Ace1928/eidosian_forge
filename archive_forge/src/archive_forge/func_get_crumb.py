from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import http_cookiejar as cookiejar
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_native
def get_crumb(module, cookies):
    resp, info = fetch_url(module, module.params['url'] + '/crumbIssuer/api/json', method='GET', timeout=module.params['timeout'], cookies=cookies)
    if info['status'] != 200:
        module.fail_json(msg='HTTP error ' + str(info['status']) + ' ' + info['msg'], output='')
    content = to_native(resp.read())
    return json.loads(content)