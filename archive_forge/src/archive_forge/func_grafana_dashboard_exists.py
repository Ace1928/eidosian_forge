from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
def grafana_dashboard_exists(module, grafana_url, uid, headers):
    dashboard_exists = False
    dashboard = {}
    grafana_version = get_grafana_version(module, grafana_url, headers)
    if grafana_version >= 5:
        uri = '%s/api/dashboards/uid/%s' % (grafana_url, uid)
    else:
        uri = '%s/api/dashboards/db/%s' % (grafana_url, uid)
    r, info = fetch_url(module, uri, headers=headers, method='GET')
    if info['status'] == 200:
        dashboard_exists = True
        try:
            dashboard = json.loads(r.read())
        except Exception as e:
            raise GrafanaAPIException(e)
    elif info['status'] == 404:
        dashboard_exists = False
    else:
        raise GrafanaAPIException('Unable to get dashboard %s : %s' % (uid, info))
    return (dashboard_exists, dashboard)