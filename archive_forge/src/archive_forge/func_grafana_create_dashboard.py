from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
def grafana_create_dashboard(module, data):
    payload = {}
    if data.get('dashboard_id'):
        data['path'] = 'https://grafana.com/api/dashboards/%s/revisions/%s/download' % (data['dashboard_id'], data['dashboard_revision'])
    if data['path'].startswith('http'):
        r, info = fetch_url(module, data['path'])
        if info['status'] != 200:
            raise GrafanaAPIException('Unable to download grafana dashboard from url %s : %s' % (data['path'], info))
        payload = json.loads(r.read())
    else:
        try:
            with open(data['path'], 'r', encoding='utf-8') as json_file:
                payload = json.load(json_file)
        except Exception as e:
            raise GrafanaAPIException("Can't load json file %s" % to_native(e))
    if 'dashboard' not in payload:
        payload = {'dashboard': payload}
    headers = grafana_headers(module, data)
    grafana_version = get_grafana_version(module, data['url'], headers)
    if grafana_version < 5:
        if data.get('slug'):
            uid = data['slug']
        elif 'meta' in payload and 'slug' in payload['meta']:
            uid = payload['meta']['slug']
        else:
            raise GrafanaMalformedJson('No slug found in json. Needed with grafana < 5')
    elif data.get('uid'):
        uid = data['uid']
    elif 'uid' in payload['dashboard']:
        uid = payload['dashboard']['uid']
    else:
        uid = None
    result = {}
    folder_exists = False
    if grafana_version >= 5:
        folder_exists, folder_id = grafana_folder_exists(module, data['url'], data['folder'], headers)
        if folder_exists is False:
            raise GrafanaAPIException("Dashboard folder '%s' does not exist." % data['folder'])
        payload['folderId'] = folder_id
    if uid:
        dashboard_exists, dashboard = grafana_dashboard_exists(module, data['url'], uid, headers=headers)
    else:
        dashboard_exists, dashboard = grafana_dashboard_search(module, data['url'], folder_id, payload['dashboard']['title'], headers=headers)
    if dashboard_exists is True:
        grafana_dashboard_changed = is_grafana_dashboard_changed(payload, dashboard)
        if grafana_dashboard_changed:
            if module.check_mode:
                module.exit_json(uid=uid, failed=False, changed=True, msg='Dashboard %s will be updated' % payload['dashboard']['title'])
            if 'overwrite' in data and data['overwrite']:
                payload['overwrite'] = True
            if 'commit_message' in data and data['commit_message']:
                payload['message'] = data['commit_message']
            r, info = fetch_url(module, '%s/api/dashboards/db' % data['url'], data=json.dumps(payload), headers=headers, method='POST')
            if info['status'] == 200:
                if grafana_version >= 5:
                    try:
                        dashboard = json.loads(r.read())
                        uid = dashboard['uid']
                    except Exception as e:
                        raise GrafanaAPIException(e)
                result['uid'] = uid
                result['msg'] = 'Dashboard %s updated' % payload['dashboard']['title']
                result['changed'] = True
            else:
                body = json.loads(info['body'])
                raise GrafanaAPIException('Unable to update the dashboard %s : %s (HTTP: %d)' % (uid, body['message'], info['status']))
        else:
            result['uid'] = uid
            result['msg'] = 'Dashboard %s unchanged.' % payload['dashboard']['title']
            result['changed'] = False
    else:
        if module.check_mode:
            module.exit_json(failed=False, changed=True, msg='Dashboard %s will be created' % payload['dashboard']['title'])
        if 'id' in payload['dashboard']:
            del payload['dashboard']['id']
        r, info = fetch_url(module, '%s/api/dashboards/db' % data['url'], data=json.dumps(payload), headers=headers, method='POST')
        if info['status'] == 200:
            result['msg'] = 'Dashboard %s created' % payload['dashboard']['title']
            result['changed'] = True
            if grafana_version >= 5:
                try:
                    dashboard = json.loads(r.read())
                    uid = dashboard['uid']
                except Exception as e:
                    raise GrafanaAPIException(e)
            result['uid'] = uid
        else:
            raise GrafanaAPIException('Unable to create the new dashboard %s : %s - %s. (headers : %s)' % (payload['dashboard']['title'], info['status'], info, headers))
    return result