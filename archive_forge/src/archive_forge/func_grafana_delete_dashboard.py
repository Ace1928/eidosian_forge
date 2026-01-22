from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
def grafana_delete_dashboard(module, data):
    headers = grafana_headers(module, data)
    grafana_version = get_grafana_version(module, data['url'], headers)
    if grafana_version < 5:
        if data.get('slug'):
            uid = data['slug']
        else:
            raise GrafanaMalformedJson('No slug parameter. Needed with grafana < 5')
    elif data.get('uid'):
        uid = data['uid']
    else:
        raise GrafanaDeleteException('No uid specified %s')
    dashboard_exists, dashboard = grafana_dashboard_exists(module, data['url'], uid, headers=headers)
    result = {}
    if dashboard_exists is True:
        if module.check_mode:
            module.exit_json(uid=uid, failed=False, changed=True, msg='Dashboard %s will be deleted' % uid)
        if grafana_version < 5:
            r, info = fetch_url(module, '%s/api/dashboards/db/%s' % (data['url'], uid), headers=headers, method='DELETE')
        else:
            r, info = fetch_url(module, '%s/api/dashboards/uid/%s' % (data['url'], uid), headers=headers, method='DELETE')
        if info['status'] == 200:
            result['msg'] = 'Dashboard %s deleted' % uid
            result['changed'] = True
            result['uid'] = uid
        else:
            raise GrafanaAPIException('Unable to update the dashboard %s : %s' % (uid, info))
    else:
        result = {'msg': 'Dashboard %s does not exist.' % uid, 'changed': False, 'uid': uid}
    return result