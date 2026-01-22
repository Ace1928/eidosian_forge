from __future__ import absolute_import, division, print_function
import json
import os
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.urls import basic_auth_header, open_url
from ansible.module_utils._text import to_native
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.utils.display import Display
def grafana_list_dashboards(self):
    headers = self.grafana_headers()
    dashboard_list = []
    try:
        if self.search:
            r = open_url('%s/api/search?query=%s' % (self.grafana_url, self.search), headers=headers, method='GET')
        else:
            r = open_url('%s/api/search/' % self.grafana_url, headers=headers, method='GET')
    except HTTPError as e:
        raise GrafanaAPIException('Unable to search dashboards : %s' % to_native(e))
    if r.getcode() == 200:
        try:
            dashboard_list = json.loads(r.read())
        except Exception as e:
            raise GrafanaAPIException('Unable to parse json list %s' % to_native(e))
    else:
        raise GrafanaAPIException('Unable to list grafana dashboards : %s' % str(r.getcode()))
    return dashboard_list