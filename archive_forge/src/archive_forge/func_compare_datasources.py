from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible_collections.community.grafana.plugins.module_utils import base
def compare_datasources(new, current, compareSecureData=True):
    if new['uid'] is None:
        del current['uid']
        del new['uid']
    del current['typeLogoUrl']
    del current['id']
    if 'version' in current:
        del current['version']
    if 'readOnly' in current:
        del current['readOnly']
    if current['basicAuth'] is False:
        if 'basicAuthUser' in current:
            del current['basicAuthUser']
    if 'password' in current:
        del current['password']
    if 'basicAuthPassword' in current:
        del current['basicAuthPassword']
    if not compareSecureData:
        new.pop('secureJsonData', None)
        new.pop('secureJsonFields', None)
        current.pop('secureJsonData', None)
        current.pop('secureJsonFields', None)
    elif not new.get('secureJsonData'):
        new.pop('secureJsonData', None)
        current.pop('secureJsonFields', None)
    else:
        current['secureJsonData'] = current.pop('secureJsonFields')
    return dict(before=current, after=new)