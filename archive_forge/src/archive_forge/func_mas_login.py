from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.basic import AnsibleModule
import codecs
def mas_login(self):
    url = '%s://%s/nitro/v1/config/login' % (self._module.params['nitro_protocol'], self._module.params['nsip'])
    login_credentials = {'login': {'username': self._module.params['nitro_user'], 'password': self._module.params['nitro_pass']}}
    data = 'object=\n%s' % self._module.jsonify(login_credentials)
    r, info = fetch_url(self._module, url=url, headers=self._headers, data=data, method='POST')
    print(r, info)
    result = {}
    self.edit_response_data(r, info, result, success_status=200)
    if result['nitro_errorcode'] == 0:
        body_data = self._module.from_json(result['http_response_body'])
        result['nitro_auth_token'] = body_data['login'][0]['sessionid']
    self._module_result['changed'] = False
    return result