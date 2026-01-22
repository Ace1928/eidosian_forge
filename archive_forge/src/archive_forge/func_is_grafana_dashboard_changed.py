from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
from ansible.module_utils._text import to_text
from ansible_collections.community.grafana.plugins.module_utils.base import (
def is_grafana_dashboard_changed(payload, dashboard):
    if 'version' in payload['dashboard']:
        del payload['dashboard']['version']
    if 'version' in dashboard['dashboard']:
        del dashboard['dashboard']['version']
    if 'meta' in dashboard:
        del dashboard['meta']
    if 'meta' in payload:
        del payload['meta']
    if 'folderId' not in dashboard:
        dashboard['folderId'] = 0
    if 'id' in dashboard['dashboard']:
        del dashboard['dashboard']['id']
    if 'id' in payload['dashboard']:
        del payload['dashboard']['id']
    if payload == dashboard:
        return False
    return True