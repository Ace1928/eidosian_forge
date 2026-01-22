from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def update_token(self, token, name, description, status, expires_at, generate_token):
    try:
        params = {}
        params['tokenid'] = token['tokenid']
        params['name'] = name
        if isinstance(description, str) and description != token['description']:
            params['description'] = description
        if isinstance(status, bool):
            if status:
                if token['status'] != '0':
                    params['status'] = '0'
            elif token['status'] != '1':
                params['status'] = '1'
        if isinstance(expires_at, int) and str(expires_at) != token['expires_at']:
            params['expires_at'] = str(expires_at)
        if len(params.keys()) == 2:
            if not generate_token:
                self._module.exit_json(changed=False)
            elif self._module.check_mode:
                self._module.exit_json(changed=True)
        else:
            if self._module.check_mode:
                self._module.exit_json(changed=True)
            self._zapi.token.update(params)
        if generate_token:
            generated_tokens = self._zapi.token.generate([token['tokenid']])
            self._module.exit_json(changed=True, msg='Successfully updated token.', token=generated_tokens[0]['token'])
        else:
            self._module.exit_json(changed=True, msg='Successfully updated token.')
    except Exception as e:
        self._module.fail_json(msg='Failed to update token: %s' % e)